use std::{ffi::c_void, mem::size_of, ops::BitAnd, sync::Arc};

use cudarc::{
    cublas::CudaBlas,
    driver::{
        result, sys::cuMemcpyDtoHAsync_v2, CudaDevice, CudaFunction, CudaSlice, DevicePtr,
        DeviceRepr, LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};
use num_traits::PrimInt;

use crate::gemm;

const PTX_SRC: &str = include_str!("kernel.cu");

pub struct MatmulEngine<T> {
    entry_size: usize,
    db_length: usize,
    query_length: usize,
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    limbs: usize,
    p: Option<T>,
    db: Vec<CudaSlice<u8>>,
    db_sums: CudaSlice<u32>,
    query_sums: CudaSlice<i32>,
    ones: CudaSlice<u8>,
    intermediate_results: CudaSlice<i32>,
    results: CudaSlice<i32>,
    function: CudaFunction,
    chunk_size: usize,
}

impl<T> MatmulEngine<T>
where
    T: PrimInt + DeviceRepr,
{
    pub fn create(
        db_entries: &[T],
        entry_size: usize,
        query_length: usize,
        chunk_size: usize,
        p: Option<T>,
    ) -> Self {
        let limbs = size_of::<T>();
        let db_length = db_entries.len() / entry_size;
        // TODO: specify device id
        let dev = CudaDevice::new(0).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();

        let ptx = compile_ptx(PTX_SRC).unwrap();
        let function_name = match (limbs, p) {
            (2, None) => "reduce_u16",
            (4, None) => "reduce_u32",
            (2, Some(_)) => "reduce_p16",
            (4, Some(_)) => "reduce_p32",
            _ => unimplemented!(),
        };

        dev.load_ptx(ptx, function_name, &[function_name]).unwrap();
        let function = dev.get_func(function_name, function_name).unwrap();

        let mut db = vec![];
        for _ in 0..limbs {
            db.push(vec![0u8; db_entries.len()]);
        }

        for (idx, entry) in db_entries.iter().enumerate() {
            for i in 0..limbs {
                let tmp = (&entry.to_u32().unwrap() >> (i * 8)) as u8;
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
        for i in 0..limbs {
            for j in 0..db_length {
                db_sums_merged[i * db_length + j] = db_sums[i][j];
            }
        }

        for i in 0..limbs {
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

        let intermediate_results_len = match p {
            Some(_) => limbs * limbs,
            None => 1,
        };

        let intermediate_results: CudaSlice<i32> = dev
            .alloc_zeros(db_length * query_length * intermediate_results_len)
            .unwrap(); // TODO

        MatmulEngine {
            entry_size,
            db_length,
            query_length,
            dev,
            blas,
            limbs,
            p,
            db,
            db_sums,
            query_sums,
            ones,
            results,
            function,
            intermediate_results,
            chunk_size,
        }
    }

    pub fn preprocess_query(&self, query: &[T]) -> Vec<Vec<u8>> {
        let mut result = vec![];
        for _ in 0..self.limbs {
            result.push(vec![0u8; query.len()]);
        }

        for (idx, entry) in query.iter().enumerate() {
            for i in 0..self.limbs {
                let tmp = (&entry.to_u32().unwrap() >> (i * 8)) as u8;
                result[i][idx] = (tmp as i32 - 128) as u8;
            }
        }

        result.to_vec()
    }

    pub fn dot(&mut self, preprocessed_query: &Vec<Vec<u8>>, results_host: *mut T) {
        let b_dev = preprocessed_query
            .iter()
            .map(|b| self.dev.htod_sync_copy(b).unwrap())
            .collect::<Vec<_>>();

        for i in 0..self.limbs {
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
            let blas = CudaBlas::new(self.dev.clone()).unwrap();
            unsafe {
                blas.set_stream(Some(&stream)).unwrap();
            }
            blass.push(blas);
            streams.push(stream);
        }

        for chunk_idx in 0..self.db_length / self.chunk_size {
            for i in 0..self.limbs {
                for j in 0..self.limbs {
                    if self.p.is_none() && i + j >= self.limbs {
                        continue;
                    }

                    gemm(
                        blass[chunk_idx].handle(),
                        &self.db[i],
                        &b_dev[j],
                        &mut self.intermediate_results,
                        (chunk_idx * self.entry_size * self.chunk_size) as u64,
                        0,
                        if self.p.is_some() {
                            ((i * self.limbs + j) * self.query_length * self.chunk_size * 4) as u64
                        } else {
                            0
                        },
                        self.chunk_size,
                        self.query_length,
                        self.entry_size,
                        if self.p.is_some() {
                            1
                        } else {
                            1 << 8 * (i + j)
                        },
                        if self.p.is_some() || (i + j == 0) {
                            0
                        } else {
                            1
                        },
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
                        self.p.unwrap_or(T::zero()),
                    ),
                )
            }
            .unwrap();

            unsafe {
                result::stream::synchronize(streams[chunk_idx].stream).unwrap();
            }

            unsafe {
                let _ = cuMemcpyDtoHAsync_v2(
                    results_host.byte_offset(
                        (self.chunk_size * chunk_idx * self.query_length * self.limbs) as isize,
                    ) as *mut c_void,
                    *self.results.device_ptr()
                        + (self.chunk_size * chunk_idx * self.query_length * self.limbs) as u64,
                    self.chunk_size * self.query_length * self.limbs,
                    streams[chunk_idx].stream,
                );
            }
        }

        for stream in streams {
            unsafe {
                result::stream::synchronize(stream.stream).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::slice;
    use std::ffi::c_void;

    use cudarc::driver::sys::cuMemAllocHost_v2;
    use ndarray::Array2;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::matmul::MatmulEngine;
    const WIDTH: usize = 12_800;
    const QUERY_SIZE: usize = 31;
    const DB_SIZE: usize = 1000;
    const CHUNK_SIZE: usize = 100;
    const RNG_SEED: u64 = 1337;

    #[test]
    // u32 ring
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

        let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, None);

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
    // u16 ring
    fn check_u16() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();
        let query = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let a_nda: ndarray::prelude::ArrayBase<
            ndarray::OwnedRepr<u32>,
            ndarray::prelude::Dim<[usize; 2]>,
        > = Array2::from_shape_vec(
            (DB_SIZE as usize, WIDTH as usize),
            db.iter().map(|x| *x as u32).collect::<Vec<_>>(),
        )
        .unwrap();

        let b_nda = Array2::from_shape_vec(
            (QUERY_SIZE as usize, WIDTH as usize),
            query.iter().map(|x| *x as u32).collect::<Vec<_>>(),
        )
        .unwrap();

        let c_nda = a_nda.dot(&b_nda.t());

        let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, None);

        let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * QUERY_SIZE * 2);
        }

        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, results_host_ptr as *mut u16);

        let gpu_result: &[u16] =
            unsafe { slice::from_raw_parts(results_host_ptr as *mut u16, DB_SIZE * QUERY_SIZE) };

        assert_eq!(
            c_nda
                .into_raw_vec()
                .iter()
                .map(|x| *x as u16)
                .collect::<Vec<_>>(),
            gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// 16 bit prime field
    fn check_p16() {
        const P: u16 = ((1u32 << 16) - 17) as u16;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();
        let query = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
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

        let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, Some(P));

        let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * QUERY_SIZE * 2);
        }

        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, results_host_ptr as *mut u16);

        let gpu_result: &[u16] =
            unsafe { slice::from_raw_parts(results_host_ptr as *mut u16, DB_SIZE * QUERY_SIZE) };

        assert_eq!(
            c_nda
                .into_raw_vec()
                .iter()
                .map(|x| (*x % (P as u64)) as u16)
                .collect::<Vec<_>>(),
            gpu_result,
            "GPU result does not match CPU implementation"
        );
    }

    #[test]
    /// 32 bit prime field
    fn check_p32() {
        const P: u32 = 4294967291;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();
        let query = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, CHUNK_SIZE, Some(P));

        let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * QUERY_SIZE * 4);
        }

        let preprocessed_query = engine.preprocess_query(&query);
        engine.dot(&preprocessed_query, results_host_ptr as *mut u32);

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
                let mut sum: u64 = 0;
                for i in 0..k {
                    sum += ((A[i + row * k] as u64) * (B[i + col * k] as u64)) % P as u64;
                }
                C[col + row * n] = (sum % P as u64) as u32;
            }
        }

        assert_eq!(
            C, gpu_result,
            "GPU result does not match CPU implementation"
        );
    }
}
