use core::slice;
use std::{ffi::c_void, sync::Arc, time::Instant};

use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas},
    driver::{
        result,
        sys::{cuMemAllocHost_v2, cuMemcpyDtoHAsync_v2, cuMemcpyDtoH_v2},
        CudaDevice, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};
use num_traits::FromPrimitive;

const PTX_SRC: &str = "
const long long INT_MAX = 4294967296;

/// Perform multiplication of two u16 in u16 ring
extern \"C\" __global__ void matmul_u16(int* c, unsigned short* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, unsigned short _p) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int a0s = a0Sums[idx % numRows];
        unsigned int a1s = a1Sums[idx % numRows];

        // Correct the sum to unsigned
        unsigned int b0s = b0Sums[idx / numRows] + numCols * 128;
        unsigned int b1s = b1Sums[idx / numRows] + numCols * 128;

        // Correct the intermediate results to unsigned
        unsigned short c00 = c[idx] + ((a0s + b0s) << 7);
        unsigned short c01 = c[idx + numElements] + ((a0s + b1s) << 7);
        unsigned short c10 = c[idx + numElements * 2] + ((a1s + b0s) << 7);

        // Calculate the u16 result
        output[idx] = c00 + ((c01 + c10) << 8);
    }
}

template<typename T>
__global__ void matmul_u32_impl(int* c, T* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, long long p) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int a0s = a0Sums[idx % numRows];
        unsigned int a1s = a1Sums[idx % numRows];

        // Correct the sum to unsigned
        int b0s = b0Sums[idx / numRows] + numCols * 128;
        int b1s = b1Sums[idx / numRows] + numCols * 128;

        // Correct the intermediate results to unsigned
        long long c00 = c[idx] + ((a0s + b0s) << 7) - (numCols * 16384);
        long long c01 = c[idx + numElements] + ((a0s + b1s) << 7) - (numCols * 16384);
        long long c10 = c[idx + numElements * 2] + ((a1s + b0s) << 7) - (numCols * 16384);
        long long c11 = c[idx + numElements * 3] + ((a1s + b1s) << 7) - (numCols * 16384);

        // Calculate the u32 result and reduce
        output[idx] = (c00 + ((c01 + c10) << 8) + (c11 << 16)) % p;
    }
}

/// Perform multiplication in 16bit field
extern \"C\" __global__ void matmul_p16(int* c, unsigned short* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, unsigned short p) {
    matmul_u32_impl<unsigned short>(c, output, a0Sums, a1Sums, b0Sums, b1Sums, numRows, numElements, numCols, static_cast<long long>(p));
}

/// Perform multiplication of two u16 in u32 ring
extern \"C\" __global__ void matmul_u16u32(int* c, unsigned int* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, unsigned short _p) {
    matmul_u32_impl<unsigned int>(c, output, a0Sums, a1Sums, b0Sums, b1Sums, numRows, numElements, numCols, INT_MAX);
}

/// Perform multiplication with Karatsuba, only for <=14bit field
extern \"C\" __global__ void matmul_p14(int* c, unsigned short* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, unsigned short p) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int a0s = a0Sums[idx % numRows];
        unsigned int a1s = a1Sums[idx % numRows];

        // Correct the sum to unsigned
        int b0s = b0Sums[idx / numRows];
        int b1s = b1Sums[idx / numRows];

        // Correct the intermediate results to unsigned
        long long c00 = c[idx];
        long long c11 = c[idx + numElements];
        long long tmp = c[idx + numElements * 2] + ((a0s + a1s + b1s + b0s) << 7) - (numCols * 16384);

        // Calculate the result and reduce
        output[idx] = (c00 + ((tmp - c00 - c11) << 7) + (c11 << 14)) % p;
    }
}

extern \"C\" __global__ void matmul_u14(unsigned int* c, unsigned short* output, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = (c[idx] + ((c[idx + numElements] + c[idx + numElements * 2]) << 7)) % 16384;
    }
}


extern \"C\" __global__ void matmul_u32(int* intermediate, int* output, unsigned int* aSums, int* bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < chunkSize*m) {
        unsigned int as[4] = {};
        unsigned int bs[4] = {};

        size_t vIdx = (idx/chunkSize) * n + (idx % chunkSize) + chunkIdx*chunkSize;
        for (int i=0;i<4;i++) {
            as[i] = aSums[(i * n) + (vIdx % n)];
            bs[i] = bSums[(i * m) + (vIdx / n)] + k * 128;
        }

        int result = intermediate[idx];
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                if ((i+j) >= 4) continue;
                result += (((as[i] + bs[j]) << 7) - (k * 16384)) << (8 * (i + j));
            }
        }

        // transpose output
        output[idx/chunkSize + (idx % chunkSize) * m + offset] = result;
    }
}

extern \"C\" __global__ void matmul_p32(int* intermediate, int* output, unsigned int* aSums, int* bSums, size_t n, size_t m, size_t k, size_t offset, size_t chunkSize, size_t chunkIdx, unsigned int p) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long pp = p;

    if (idx < chunkSize*m) {
        unsigned int as[4] = {};
        unsigned int bs[4] = {};

        size_t vIdx = (idx/chunkSize) * n + (idx % chunkSize) + chunkIdx*chunkSize;
        for (int i=0;i<4;i++) {
            as[i] = aSums[(i * n) + (vIdx % n)];
            bs[i] = bSums[(i * m) + (vIdx / n)] + k * 128;
        }

        int result[4] = {};
        for (int i=0;i<4;i++) {
            result[i] = intermediate[idx + chunkSize * m * i];
        }

        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                size_t ii = (i % 2);
                size_t jj = (j % 2);
                size_t result_idx = (i / 2) * 2 + (j / 2);
                result[result_idx] += (((as[i] + bs[j]) << 7) - (k * 16384)) << (8 * (ii + jj));
            }
        }

        unsigned long long final_result = 0;
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                unsigned long long f = (1 << (16*(i+j))) % pp;
                final_result += ((result[i*2+j] % pp) * f) % pp;
            }
        }

        // transpose output
        output[idx/chunkSize + (idx % chunkSize) * m + offset] = final_result % pp;
    }
}
";

pub fn gemm(
    handle: &sys::cublasHandle_t,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<i32>,
    a_offset: u64,
    b_offset: u64,
    c_offset: u64,
    m: usize,
    n: usize,
    k: usize,
    alpha: i32,
    beta: i32,
) {
    unsafe {
        gemm_ex(
            handle.clone(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const i32 as *const c_void,
            (*a.device_ptr() + a_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            (*b.device_ptr() + b_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            &beta as *const i32 as *const c_void,
            (*c.device_ptr_mut() + c_offset) as *mut _,
            sys::cublasDataType_t::CUDA_R_32I,
            m as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I_PEDANTIC,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .unwrap();
    }
}

#[derive(PartialEq)]
pub enum ComputeDataType {
    P14,
    U14,
    U16,
    P16,
    U16U32,
    U32,
    P32,
}

pub struct MatmulEngineU16<T> {
    entry_size: usize,
    db_length: usize,
    query_length: usize,
    blas: CudaBlas,
    dev: Arc<CudaDevice>,
    db1: CudaSlice<u8>,
    db0: CudaSlice<u8>,
    db01: CudaSlice<u8>,
    db1_sums: CudaSlice<u32>,
    db0_sums: CudaSlice<u32>,
    query1_sums: CudaSlice<i32>,
    query0_sums: CudaSlice<i32>,
    ones: CudaSlice<u8>,
    intermediate_results: CudaSlice<i32>,
    results: CudaSlice<T>,
    function: CudaFunction,
    data_type: ComputeDataType,
    p: u16,
}

impl<T> MatmulEngineU16<T>
where
    T: FromPrimitive
        + Copy
        + std::iter::Sum
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + std::default::Default,
{
    pub fn create(
        db_entries: &[u16],
        entry_size: usize,
        query_length: usize,
        data_type: ComputeDataType,
        p: Option<u16>,
    ) -> Self {
        let db_length = db_entries.len() / entry_size;
        // TODO: specify device id
        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();

        let function_name = match data_type {
            ComputeDataType::U16 => "matmul_u16",
            ComputeDataType::P16 => "matmul_p16",
            ComputeDataType::U16U32 => "matmul_u16u32",
            ComputeDataType::P14 => "matmul_p14",
            ComputeDataType::U14 => "matmul_u14",
            _ => todo!(),
        };

        let ptx = compile_ptx(PTX_SRC).unwrap();
        dev.load_ptx(ptx, function_name, &[function_name]).unwrap();
        let function = dev.get_func(function_name, function_name).unwrap();

        let (mask1, mask0, offset) = match data_type {
            ComputeDataType::P14 | ComputeDataType::U14 => (7, 0x7F, 0),
            _ => (8, 0xFF, 128),
        };

        let mut a1_host = db_entries
            .iter()
            .map(|x: &u16| (x >> mask1) as u8)
            .collect::<Vec<_>>();
        let mut a0_host = db_entries
            .iter()
            .map(|x| (x & mask0) as u8)
            .collect::<Vec<_>>();

        let db01 = if data_type == ComputeDataType::P14 {
            let a01_host: Vec<u8> = a1_host
                .iter()
                .zip(a0_host.iter())
                .map(|(&a1, &a0)| ((a1 + a0) as i32 - 128) as u8)
                .collect();

            dev.htod_sync_copy(&a01_host).unwrap()
        } else {
            dev.htod_sync_copy(&[0u8; 0]).unwrap()
        };

        let a1_sums: Vec<u32> = a1_host
            .chunks(entry_size)
            .map(|row| row.iter().map(|&x| x as u32).sum())
            .collect();
        let a0_sums: Vec<u32> = a0_host
            .chunks(entry_size)
            .map(|row| row.iter().map(|&x| x as u32).sum())
            .collect();

        a1_host
            .iter_mut()
            .for_each(|x| (*x = (*x as i32 - offset) as u8));

        a0_host
            .iter_mut()
            .for_each(|x| (*x = (*x as i32 - offset) as u8));

        let db1 = dev.htod_sync_copy(&a1_host).unwrap();
        let db0 = dev.htod_sync_copy(&a0_host).unwrap();
        let db1_sums = dev.htod_sync_copy(&a1_sums).unwrap();
        let db0_sums = dev.htod_sync_copy(&a0_sums).unwrap();

        let query1_sums: CudaSlice<i32> = dev.alloc_zeros(query_length).unwrap();
        let query0_sums: CudaSlice<i32> = dev.alloc_zeros(query_length).unwrap();

        let ones = vec![1u8; entry_size];
        let ones = dev.htod_sync_copy(&ones).unwrap();

        let results: CudaSlice<T> = dev.alloc_zeros(db_length * query_length).unwrap();

        let intermediate_results_count = match data_type {
            ComputeDataType::U14 | ComputeDataType::P14 | ComputeDataType::U16 => 3,
            ComputeDataType::P16 | ComputeDataType::U16U32 => 4,
            _ => todo!(),
        };

        let intermediate_results: CudaSlice<i32> = dev
            .alloc_zeros(db_length * query_length * intermediate_results_count)
            .unwrap();

        MatmulEngineU16 {
            entry_size,
            db_length,
            query_length,
            blas,
            dev,
            db1,
            db0,
            db01,
            db1_sums,
            db0_sums,
            query1_sums,
            query0_sums,
            ones,
            results,
            function,
            data_type,
            intermediate_results,
            p: p.unwrap_or(0),
        }
    }

    pub fn preprocess_query(&self, query: &[u16]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        match self.data_type {
            ComputeDataType::P14 => {
                let (b1, b0, b01) = self.prepare_query_karatsuba(query);
                (b1, b0, b01)
            }
            ComputeDataType::U14 => {
                let (b1, b0) = self.prepare_query_u14(query);
                (b1, b0, vec![])
            }
            _ => {
                let (b1, b0) = self.prepare_query(query);
                (b1, b0, vec![])
            }
        }
    }

    pub fn dot(
        &mut self,
        preprocessed_query: &(Vec<u8>, Vec<u8>, Vec<u8>),
        results_host: &mut Vec<T>,
    ) {
        let (b1, b0, b01) = preprocessed_query;

        let b1_dev = self.dev.htod_sync_copy(&b1).unwrap();
        let b0_dev = self.dev.htod_sync_copy(&b0).unwrap();

        // Calculate row sums for sign correction
        if self.data_type != ComputeDataType::U14 {
            gemm(
                &self.blas.handle(),
                &b1_dev,
                &self.ones,
                &mut self.query1_sums,
                0,
                0,
                0,
                self.query_length,
                1,
                self.entry_size,
                1,
                0,
            );

            gemm(
                &self.blas.handle(),
                &b0_dev,
                &self.ones,
                &mut self.query0_sums,
                0,
                0,
                0,
                self.query_length,
                1,
                self.entry_size,
                1,
                0,
            );
        }

        // Calculate byte-wise products
        gemm(
            &self.blas.handle(),
            &self.db0,
            &b0_dev,
            &mut self.intermediate_results,
            0,
            0,
            0,
            self.db_length,
            self.query_length,
            self.entry_size,
            1,
            0,
        );

        if self.data_type == ComputeDataType::P14 {
            // Use Karatsuba for P14
            gemm(
                &self.blas.handle(),
                &self.db1,
                &b1_dev,
                &mut self.intermediate_results,
                0,
                0,
                (self.db_length * self.query_length * 4 * 1) as u64,
                self.db_length,
                self.query_length,
                self.entry_size,
                1,
                0,
            );

            let b01_dev = self.dev.htod_sync_copy(&b01).unwrap();

            gemm(
                &self.blas.handle(),
                &self.db01,
                &b01_dev,
                &mut self.intermediate_results,
                0,
                0,
                (self.db_length * self.query_length * 4 * 2) as u64,
                self.db_length,
                self.query_length,
                self.entry_size,
                1,
                0,
            );
        } else {
            // Default byte-wise impl for the rest
            gemm(
                &self.blas.handle(),
                &self.db0,
                &b1_dev,
                &mut self.intermediate_results,
                0,
                0,
                (self.db_length * self.query_length * 4 * 1) as u64,
                self.db_length,
                self.query_length,
                self.entry_size,
                1,
                0,
            );

            gemm(
                &self.blas.handle(),
                &self.db1,
                &b0_dev,
                &mut self.intermediate_results,
                0,
                0,
                (self.db_length * self.query_length * 4 * 2) as u64,
                self.db_length,
                self.query_length,
                self.entry_size,
                1,
                0,
            );

            // Additional matmul needed with high bytes for u32
            if (self.data_type == ComputeDataType::P16)
                | (self.data_type == ComputeDataType::U16U32)
            {
                gemm(
                    &self.blas.handle(),
                    &self.db1,
                    &b1_dev,
                    &mut self.intermediate_results,
                    0,
                    0,
                    (self.db_length * self.query_length * 4 * 3) as u64,
                    self.db_length,
                    self.query_length,
                    self.entry_size,
                    1,
                    0,
                );
            }
        }

        let num_elements = self.db_length * self.query_length;
        let threads_per_block = 256;
        let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            if self.data_type == ComputeDataType::U14 {
                self.function.clone().launch(
                    cfg,
                    (
                        &self.intermediate_results,
                        &mut self.results,
                        (self.db_length * self.query_length) as u64,
                    ),
                )
            } else {
                self.function.clone().launch(
                    cfg,
                    (
                        &self.intermediate_results,
                        &mut self.results,
                        &self.db0_sums,
                        &self.db1_sums,
                        &self.query0_sums,
                        &self.query1_sums,
                        self.db_length as u64,
                        (self.db_length * self.query_length) as u64,
                        self.entry_size as u64,
                        self.p,
                    ),
                )
            }
        }
        .unwrap();

        self.dev
            .dtoh_sync_copy_into(&self.results, results_host)
            .unwrap();
    }

    pub fn prepare_query(&self, query: &[u16]) -> (Vec<u8>, Vec<u8>) {
        let mut b1 = vec![0u8; query.len()];
        let mut b0 = vec![0u8; query.len()];

        for i in 0..query.len() {
            b1[i] = ((query[i] >> 8) as i32 - 128) as u8;
            b0[i] = ((query[i] & 0xFF) as i32 - 128) as u8;
        }

        (b1, b0)
    }

    pub fn prepare_query_u14(&self, query: &[u16]) -> (Vec<u8>, Vec<u8>) {
        let mut b1 = vec![0u8; query.len()];
        let mut b0 = vec![0u8; query.len()];

        for i in 0..query.len() {
            b1[i] = (query[i] >> 7) as u8;
            b0[i] = (query[i] & 0x7F) as u8;
        }

        (b1, b0)
    }

    pub fn prepare_query_karatsuba(&self, query: &[u16]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut b1 = vec![0u8; query.len()];
        let mut b0 = vec![0u8; query.len()];
        let mut b01 = vec![0u8; query.len()];

        for i in 0..query.len() {
            let tmp_1 = query[i] >> 7;
            let tmp_0 = query[i] & 0x7F;
            b1[i] = tmp_1 as u8;
            b0[i] = tmp_0 as u8;
            b01[i] = ((tmp_1 + tmp_0) as i8 - 127 - 1) as u8;
        }

        (b1, b0, b01)
    }
}

/// U32

pub struct MatmulEngineU32 {
    entry_size: usize,
    db_length: usize,
    query_length: usize,
    blas: CudaBlas,
    dev: Arc<CudaDevice>,
    db: Vec<CudaSlice<u8>>,
    db_sums: CudaSlice<u32>,
    query_sums: CudaSlice<i32>,
    ones: CudaSlice<u8>,
    intermediate_results: CudaSlice<i32>,
    results: CudaSlice<i32>,
    function: CudaFunction,
    data_type: ComputeDataType,
    chunk_size: usize,
}

impl MatmulEngineU32 {
    pub fn create(
        db_entries: &[u32],
        entry_size: usize,
        query_length: usize,
        chunk_size: usize,
        data_type: ComputeDataType,
    ) -> Self {
        let db_length = db_entries.len() / entry_size;
        // TODO: specify device id
        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();

        let ptx = compile_ptx(PTX_SRC).unwrap();
        let function_name = match data_type {
            ComputeDataType::U32 => "matmul_u32",
            ComputeDataType::P32 => "matmul_p32",
            _ => unimplemented!(),
        };
        dev.load_ptx(ptx, function_name, &[function_name]).unwrap();
        let function = dev.get_func(function_name, function_name).unwrap();

        let mut db = [
            vec![0u8; db_entries.len()],
            vec![0u8; db_entries.len()],
            vec![0u8; db_entries.len()],
            vec![0u8; db_entries.len()],
        ];

        for (idx, entry) in db_entries.iter().enumerate() {
            for i in 0..4 {
                let tmp = (entry >> (i * 8)) as u8;
                db[i][idx] = tmp;
            }
        }

        let db_sums = db
            .iter()
            .map(|partial_db| {
                partial_db
                    .chunks(entry_size)
                    .map(|row| row.iter().map(|&x| x as u32).sum())
                    .collect()
            })
            .collect::<Vec<Vec<u32>>>();

        let mut db_sums_merged = vec![0u32; db_length * 4];
        for i in 0..4 {
            for j in 0..db_length {
                db_sums_merged[i * db_length + j] = db_sums[i][j];
            }
        }

        for i in 0..4 {
            for j in 0..db[i].len() {
                db[i][j] = (db[i][j] as i32 - 128) as u8;
            }
        }

        let db = db
            .iter()
            .map(|partial_db| dev.htod_sync_copy(partial_db).unwrap())
            .collect::<Vec<_>>();

        let db_sums = dev.htod_sync_copy(&db_sums_merged).unwrap();
        let query_sums: CudaSlice<i32> = dev.alloc_zeros(query_length * 4).unwrap();

        let ones = vec![1u8; entry_size];
        let ones = dev.htod_sync_copy(&ones).unwrap();

        let results: CudaSlice<i32> = dev.alloc_zeros(db_length * query_length).unwrap();

        let intermediate_results: CudaSlice<i32> =
            dev.alloc_zeros(db_length * query_length * 14).unwrap(); // TODO

        MatmulEngineU32 {
            entry_size,
            db_length,
            query_length,
            blas,
            dev,
            db,
            db_sums,
            query_sums,
            ones,
            results,
            function,
            data_type,
            intermediate_results,
            chunk_size,
        }
    }

    pub fn preprocess_query(&self, query: &[u32]) -> Vec<Vec<u8>> {
        let mut result = [
            vec![0u8; query.len()],
            vec![0u8; query.len()],
            vec![0u8; query.len()],
            vec![0u8; query.len()],
        ];

        for (idx, entry) in query.iter().enumerate() {
            for i in 0..4 {
                let tmp = (entry >> (i * 8)) as u8;
                result[i][idx] = (tmp as i32 - 128) as u8;
            }
        }

        result.to_vec()
    }

    pub fn dot(&mut self, preprocessed_query: &Vec<Vec<u8>>, results_host: *mut u32) {
        let b_dev = preprocessed_query
            .iter()
            .map(|b| self.dev.htod_sync_copy(b).unwrap())
            .collect::<Vec<_>>();

        for i in 0..4 {
            gemm(
                &self.blas.handle(),
                &b_dev[i],
                &self.ones,
                &mut self.query_sums,
                0,
                0,
                (i * self.query_length * 4) as u64,
                self.query_length,
                1,
                self.entry_size,
                1,
                0,
            );
        }

        let num_elements = self.chunk_size * self.query_length;
        let threads_per_block = 256;
        let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut blass = vec![];
        let mut streams = vec![];

        for _ in 0..self.db_length / self.chunk_size {
            let stream = self.dev.fork_default_stream().unwrap();
            let mut blas = CudaBlas::new(self.dev.clone()).unwrap();
            unsafe {
                blas.set_stream(Some(&stream));
            }
            blass.push(blas);
            streams.push(stream);
        }

        for chunk_idx in 0..self.db_length / self.chunk_size {
            for i in 0..4 {
                for j in 0..4 {
                    if i + j >= 4 {
                        continue;
                    }

                    gemm(
                        blass[chunk_idx].handle(),
                        &self.db[i],
                        &b_dev[j],
                        &mut self.intermediate_results,
                        (chunk_idx * self.entry_size * self.chunk_size) as u64,
                        0,
                        0,
                        //(chunk_idx * self.query_length * self.chunk_size * 4) as u64,
                        self.chunk_size,
                        self.query_length,
                        self.entry_size,
                        (1 << 8 * (i + j)) as i32,
                        if i + j == 0 { 0 } else { 1 },
                    );
                }
            }

            unsafe {
                self.function.clone().launch_on_stream(
                    &streams[chunk_idx],
                    cfg,
                    (
                        &self.intermediate_results,
                        &mut self.results,
                        &self.db_sums,
                        &self.query_sums,
                        self.db_length as u64,
                        self.query_length as u64,
                        self.entry_size as u64,
                        (chunk_idx * self.chunk_size * self.query_length) as u64,
                        self.chunk_size as u64,
                        chunk_idx as u64,
                    ),
                )
            }
            .unwrap();

            unsafe {
                result::stream::synchronize(streams[chunk_idx].stream);
            }

            unsafe {
                let _ = cuMemcpyDtoHAsync_v2(
                    results_host
                        .byte_offset((self.chunk_size * chunk_idx * self.query_length * 4) as isize)
                        as *mut c_void,
                    *self.results.device_ptr()
                        + (self.chunk_size * chunk_idx * self.query_length * 4) as u64,
                    self.chunk_size * self.query_length * 4,
                    streams[chunk_idx].stream,
                );
            }
        }

        for stream in streams {
            unsafe {
                result::stream::synchronize(stream.stream);
            }
        }
    }

    /// p32 x p32 → p32
    pub fn dot_p32(&mut self, preprocessed_query: &Vec<Vec<u8>>, results_host: *mut u32) {
        let b_dev = preprocessed_query
            .iter()
            .map(|b| self.dev.htod_sync_copy(b).unwrap())
            .collect::<Vec<_>>();

        for i in 0..4 {
            gemm(
                &self.blas.handle(),
                &b_dev[i],
                &self.ones,
                &mut self.query_sums,
                0,
                0,
                (i * self.query_length * 4) as u64,
                self.query_length,
                1,
                self.entry_size,
                1,
                0,
            );
        }

        let num_elements = self.chunk_size * self.query_length;
        let threads_per_block = 256;
        let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut blass = vec![];
        let mut streams = vec![];

        for _ in 0..self.db_length / self.chunk_size {
            let stream = self.dev.fork_default_stream().unwrap();
            let mut blas = CudaBlas::new(self.dev.clone()).unwrap();
            unsafe {
                blas.set_stream(Some(&stream));
            }
            blass.push(blas);
            streams.push(stream);
        }

        for chunk_idx in 0..self.db_length / self.chunk_size {
            for i in 0..4 {
                for j in 0..4 {
                    let ii = i % 2;
                    let jj = j % 2;
                    gemm(
                        blass[chunk_idx].handle(),
                        &self.db[i],
                        &b_dev[j],
                        &mut self.intermediate_results,
                        (chunk_idx * self.entry_size * self.chunk_size) as u64,
                        0,
                        (((i/2) * 2 + (j/2)) * self.query_length * self.chunk_size * 4) as u64,
                        self.chunk_size,
                        self.query_length,
                        self.entry_size,
                        (1 << 8 * (ii + jj)) as i32,
                        if ii + jj == 0 { 0 } else { 1 },
                    );
                }
            }

            unsafe {
                self.function.clone().launch_on_stream(
                    &streams[chunk_idx],
                    cfg,
                    (
                        &self.intermediate_results,
                        &mut self.results,
                        &self.db_sums,
                        &self.query_sums,
                        self.db_length as u64,
                        self.query_length as u64,
                        self.entry_size as u64,
                        (chunk_idx * self.chunk_size * self.query_length) as u64,
                        self.chunk_size as u64,
                        chunk_idx as u64,
                        1337,
                    ),
                )
            }
            .unwrap();

            unsafe {
                result::stream::synchronize(streams[chunk_idx].stream);
            }

            unsafe {
                let _ = cuMemcpyDtoHAsync_v2(
                    results_host
                        .byte_offset((self.chunk_size * chunk_idx * self.query_length * 4) as isize)
                        as *mut c_void,
                    *self.results.device_ptr()
                        + (self.chunk_size * chunk_idx * self.query_length * 4) as u64,
                    self.chunk_size * self.query_length * 4,
                    streams[chunk_idx].stream,
                );
            }
        }

        for stream in streams {
            unsafe {
                result::stream::synchronize(stream.stream);
            }
        }
    }
}

#[cfg(test)]
/// Sanity checks for correctness
mod tests {
    use core::slice;
    use std::ffi::c_void;

    use cudarc::driver::sys::cuMemAllocHost_v2;
    use ndarray::Array2;
    use num_traits::FromPrimitive;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{ComputeDataType, MatmulEngineU16, MatmulEngineU32};
    const WIDTH: usize = 12_800;
    const QUERY_SIZE: usize = 31;
    const DB_SIZE: usize = 1000;
    const CHUNK_SIZE: usize = 100;
    const RNG_SEED: u64 = 1337;

    /// Helpers
    fn random_ndarray<T>(array: Vec<u16>, n: usize, m: usize) -> Array2<T>
    where
        T: FromPrimitive,
    {
        Array2::from_shape_vec(
            (n as usize, m as usize),
            array
                .into_iter()
                .map(|x| T::from_u16(x).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn random_vec(n: usize, m: usize, max_value: u32) -> Vec<u16> {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        (0..n * m)
            .map(|_| rng.gen_range(0..max_value) as u16)
            .collect()
    }

    #[test]
    /// u16 x u16 → u16
    fn check_u16() {
        let db = random_vec(DB_SIZE, WIDTH, 1 << 16);
        let query = random_vec(QUERY_SIZE, WIDTH, 1 << 16);
        let mut gpu_result = vec![0u16; DB_SIZE * QUERY_SIZE];

        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U16, None);
        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, &mut gpu_result);

        let a_nda = random_ndarray::<u16>(db, DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u16>(query, QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push(*row);
            }
        }

        assert_eq!(
            vec_column_major, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// p16 x p16 → p16
    fn check_p16() {
        const P: u16 = ((1u32 << 16) - 17) as u16;

        let db = random_vec(DB_SIZE, WIDTH, P as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
        let mut gpu_result = vec![0u16; DB_SIZE * QUERY_SIZE];

        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::P16, Some(P));
        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, &mut gpu_result);

        let a_nda = random_ndarray::<u64>(db, DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u64>(query, QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push((*row % (P as u64)) as u16);
            }
        }

        assert_eq!(
            vec_column_major, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// u16 x u16 → u32
    fn check_u16u32() {
        let db = random_vec(DB_SIZE, WIDTH, 1 << 16);
        let query = random_vec(QUERY_SIZE, WIDTH, 1 << 16);
        let mut gpu_result = vec![0u32; DB_SIZE * QUERY_SIZE];

        let mut engine =
            MatmulEngineU16::<u32>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U16U32, None);
        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, &mut gpu_result);

        let a_nda = random_ndarray::<u64>(db, DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u64>(query, QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u32> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push(*row as u32);
            }
        }

        assert_eq!(
            vec_column_major, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// p14 x p14 → p14
    fn check_p14() {
        const P: u16 = (1 << 14) - 3;

        let db = random_vec(DB_SIZE, WIDTH, P as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
        let mut gpu_result = vec![0u16; DB_SIZE * QUERY_SIZE];

        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::P14, Some(P));
        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, &mut gpu_result);

        let a_nda = random_ndarray::<u64>(db, DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u64>(query, QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push((*row % (P as u64)) as u16);
            }
        }

        assert_eq!(
            vec_column_major, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// u14 x u14 → u14
    fn check_u14() {
        let db = random_vec(DB_SIZE, WIDTH, 1 << 14);
        let query = random_vec(QUERY_SIZE, WIDTH, 1 << 14);
        let mut gpu_result = vec![0u16; DB_SIZE * QUERY_SIZE];

        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U14, None);
        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, &mut gpu_result);

        let a_nda = random_ndarray::<u16>(db, DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u16>(query, QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push((*row % (1 << 14)) as u16);
            }
        }

        assert_eq!(
            vec_column_major, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// u32 x u32 → u32
    fn check_u32() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u32>())
            .collect::<Vec<_>>();
        let query = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen::<u32>())
            .collect::<Vec<_>>();

        let a_nda: ndarray::prelude::ArrayBase<
            ndarray::OwnedRepr<u64>,
            ndarray::prelude::Dim<[usize; 2]>,
        > = Array2::from_shape_vec(
            (DB_SIZE as usize, WIDTH as usize),
            db.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        )
        .unwrap();

        let b_nda = Array2::from_shape_vec(
            (QUERY_SIZE as usize, WIDTH as usize),
            query.iter().map(|x| *x as u64).collect::<Vec<_>>(),
        )
        .unwrap();

        let c_nda = a_nda.dot(&b_nda.t());

        let mut engine =
            MatmulEngineU32::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, ComputeDataType::U32);

        let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * QUERY_SIZE * 4);
        }

        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, results_host_ptr as *mut u32);

        let gpu_result: &[u32] =
            unsafe { slice::from_raw_parts(results_host_ptr as *mut u32, DB_SIZE * QUERY_SIZE) };

        assert_eq!(
            c_nda
                .into_raw_vec()
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>(),
            gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// p32 x p32 → p32
    fn check_p32() {
        const p: u32 = 1337;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH).map(|_| 300).collect::<Vec<_>>();
        let query = (0..QUERY_SIZE * WIDTH).map(|_| 2).collect::<Vec<_>>();

        let mut engine =
            MatmulEngineU32::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, ComputeDataType::P32);

        let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * QUERY_SIZE * 4);
        }

        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot_p32(&preprocessed_query, results_host_ptr as *mut u32);

        let gpu_result: &[u32] =
            unsafe { slice::from_raw_parts(results_host_ptr as *mut u32, DB_SIZE * QUERY_SIZE) };

        let a_nda = Array2::from_shape_vec(
            (DB_SIZE as usize, WIDTH as usize),
            db.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
        )
        .unwrap();

        let b_nda = Array2::from_shape_vec(
            (QUERY_SIZE as usize, WIDTH as usize),
            query.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
        )
        .unwrap();

        let m = DB_SIZE;
        let n = QUERY_SIZE;
        let k = WIDTH;
        let A = a_nda.into_raw_vec();
        let B = b_nda.into_raw_vec();
        let mut C = vec![0u32; n * m];

        for row in 0..m {
            for col in 0..n {
                let mut sum: u32 = 0;
                for i in 0..k {
                    sum += (A[i + row * k] * B[i + col * k]) % p;
                }
                C[row + col * m] = sum % p;
            }
        }

        assert_eq!(
            C[0..100],
            gpu_result[0..100],
            "GPU result does not match CPU implementation"
        );
    }
}
