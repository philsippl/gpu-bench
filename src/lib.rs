use std::{default, ffi::c_void, sync::Arc};

use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas},
    driver::{
        CudaDevice, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};

const PTX_SRC: &str = "
extern \"C\" __global__ void calc_u16(int* c, unsigned short* output, unsigned short* a0Sums, unsigned short* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned short a0s = a0Sums[idx % numRows];
        unsigned short a1s = a1Sums[idx % numRows];
        unsigned int b0s = b0Sums[idx / numRows] + numCols * 128;
        unsigned int b1s = b1Sums[idx / numRows] + numCols * 128;

        // Correct the results to simulate u8
        unsigned short c00 = c[idx];
        c00 += (a0s + b0s) << 7;

        unsigned short c01 = c[idx + numElements];
        c01 += (a0s + b1s) << 7;

        unsigned short c10 = c[idx + numElements * 2];
        c10 += (a1s + b0s) << 7;

        // Calculate the u16 result
        output[idx] = c00 + ((c01 + c10) << 8);
    }
}
";

fn gemm(
    handle: &sys::cublasHandle_t,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<i32>,
    c_offset: u64,
    m: i32,
    n: i32,
    k: i32,
) {
    unsafe {
        gemm_ex(
            handle.clone(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            k as i32,
            &1 as *const i32 as *const c_void,
            *a.device_ptr() as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            *b.device_ptr() as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            &0 as *const i32 as *const c_void,
            (*c.device_ptr_mut() + c_offset) as *mut _,
            sys::cublasDataType_t::CUDA_R_32I,
            m as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .unwrap();
    }
}

// pub trait GPUDotEngine {
//     fn create_database(&mut self, db_entries: &[u16], entry_size: usize, query_length: usize);
//     fn dot(&self, query: &[u16]) -> &[u16];
// }

pub struct U16RingEngine {
    entry_size: usize,
    db_length: usize,
    query_length: usize,
    blas: CudaBlas,
    dev: Arc<CudaDevice>,
    db1: CudaSlice<u8>,
    db0: CudaSlice<u8>,
    db1_sums: CudaSlice<u16>,
    db0_sums: CudaSlice<u16>,
    query1_sums: CudaSlice<i32>,
    query0_sums: CudaSlice<i32>,
    ones: CudaSlice<u8>,
    results: CudaSlice<u16>,
    function: CudaFunction,
}

impl U16RingEngine {
    pub fn create(db_entries: &[u16], entry_size: usize, query_length: usize) -> Self {
        let db_length = db_entries.len() / entry_size;
        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();

        let ptx = compile_ptx(PTX_SRC).unwrap();
        dev.load_ptx(ptx, "calc_u16", &["calc_u16"]).unwrap();
        let function = dev.get_func("calc_u16", "calc_u16").unwrap();

        let mut a1_host = db_entries
            .iter()
            .map(|x: &u16| (x >> 8) as u8)
            .collect::<Vec<_>>();
        let a1_sums: Vec<u16> = a1_host
            .chunks(entry_size)
            .map(|row| row.iter().map(|&x| x as u32).sum::<u32>() as u16)
            .collect();
        a1_host
            .iter_mut()
            .for_each(|x| (*x = (*x as i8 - 127 - 1) as u8));

        let mut a0_host = db_entries
            .iter()
            .map(|x| (x & 0xFF) as u8)
            .collect::<Vec<_>>();
        let a0_sums: Vec<u16> = a0_host
            .chunks(entry_size)
            .map(|row| row.iter().map(|&x| x as u32).sum::<u32>() as u16)
            .collect();
        a0_host
            .iter_mut()
            .for_each(|x| (*x = (*x as i8 - 127 - 1) as u8));

        let db1 = dev.htod_sync_copy(&a1_host).unwrap();
        let db0 = dev.htod_sync_copy(&a0_host).unwrap();
        let db1_sums = dev.htod_sync_copy(&a1_sums).unwrap();
        let db0_sums = dev.htod_sync_copy(&a0_sums).unwrap();

        let query1_sums: CudaSlice<i32> = dev.alloc_zeros(query_length * 4).unwrap();
        let query0_sums: CudaSlice<i32> = dev.alloc_zeros(query_length * 4).unwrap();

        let ones = vec![1u8; entry_size];
        let ones = dev.htod_sync_copy(&ones).unwrap();

        let results_host = vec![0u16; db_length * query_length];
        let results = dev.htod_sync_copy(&results_host).unwrap();

        U16RingEngine {
            entry_size,
            db_length,
            query_length,
            blas,
            dev,
            db1,
            db0,
            db1_sums,
            db0_sums,
            query1_sums,
            query0_sums,
            ones,
            results,
            function,
        }
    }

    fn dot(&mut self, query: &[u16]) -> Vec<u16> {
        let (b1, b0) = self.prepare_query(query);
        let b1_dev = self.dev.htod_sync_copy(&b1).unwrap();
        let b0_dev = self.dev.htod_sync_copy(&b0).unwrap();

        let mut c_dev = self
            .dev
            .alloc_zeros(self.db_length * self.query_length * 3 * 4)
            .unwrap();

        gemm(
            &self.blas.handle(),
            &b1_dev,
            &self.ones,
            &mut self.query1_sums,
            0,
            self.query_length as i32,
            1,
            self.entry_size as i32,
        );

        gemm(
            &self.blas.handle(),
            &b0_dev,
            &self.ones,
            &mut self.query0_sums,
            0,
            self.query_length as i32,
            1,
            self.entry_size as i32,
        );

        gemm(
            &self.blas.handle(),
            &self.db0,
            &b0_dev,
            &mut c_dev,
            0,
            self.db_length as i32,
            self.query_length as i32,
            self.entry_size as i32,
        );
        gemm(
            &self.blas.handle(),
            &self.db0,
            &b1_dev,
            &mut c_dev,
            (self.db_length * self.query_length * 4 * 1) as u64,
            self.db_length as i32,
            self.query_length as i32,
            self.entry_size as i32,
        );
        gemm(
            &self.blas.handle(),
            &self.db1,
            &b0_dev,
            &mut c_dev,
            (self.db_length * self.query_length * 4 * 2) as u64,
            self.db_length as i32,
            self.query_length as i32,
            self.entry_size as i32,
        );

        let num_elements = self.db_length * self.query_length;
        let threads_per_block = 256;
        let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.function.clone().launch(
                cfg,
                (
                    &c_dev,
                    &mut self.results,
                    &self.db0_sums,
                    &self.db1_sums,
                    &self.query0_sums,
                    &self.query1_sums,
                    self.db_length as u64,
                    (self.db_length * self.query_length) as u64,
                    self.entry_size as u64,
                ),
            )
        }
        .unwrap();

        let mut results_host = vec![0u16; self.db_length * self.query_length];
        self.dev
            .dtoh_sync_copy_into(&self.results, &mut results_host)
            .unwrap();

        results_host
    }

    pub fn prepare_query(&self, query: &[u16]) -> (Vec<u8>, Vec<u8>) {
        let mut b1 = vec![0u8; query.len()];
        let mut b0 = vec![0u8; query.len()];

        for i in 0..query.len() {
            b1[i] = ((query[i] >> 8) as i8 - 127 - 1) as u8;
            b0[i] = ((query[i] & 0xFF) as i8 - 127 - 1) as u8;
        }

        (b1, b0)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::U16RingEngine;

    #[test]
    fn check_equivalence() {
        const WIDTH: usize = 12_800;
        const QUERY_SIZE: usize = 31;
        const DB_SIZE: usize = 1000;
        const RNG_SEED: u64 = 40;

        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let query = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let mut engine = U16RingEngine::create(&db, WIDTH, QUERY_SIZE);

        let gpu_result = engine.dot(&query);

        let a_nda = Array2::from_shape_vec(
            (DB_SIZE as usize, WIDTH as usize),
            db.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
        )
        .unwrap();
        let b_nda = Array2::from_shape_vec(
            (QUERY_SIZE as usize, WIDTH as usize),
            query.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
        )
        .unwrap();
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
}
