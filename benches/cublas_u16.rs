use std::ffi::c_void;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cudarc::cublas::result::gemm_ex;
use cudarc::cublas::{sys, CudaBlas};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 32;
const DB_SIZE: usize = 100_000;
const RNG_SEED: u64 = 42;

fn gemm(
    handle: &sys::cublasHandle_t,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u32>,
    c_offset: u64,
) {
    unsafe {
        gemm_ex(
            handle.clone(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            DB_SIZE as i32,
            QUERY_SIZE as i32,
            WIDTH as i32,
            &1 as *const i32 as *const c_void,
            *a.device_ptr() as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            WIDTH as i32,
            *b.device_ptr() as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            WIDTH as i32,
            &0 as *const i32 as *const c_void,
            (*c.device_ptr_mut() + c_offset) as *mut _,
            sys::cublasDataType_t::CUDA_R_32I,
            DB_SIZE as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .unwrap();
    }
}

fn cublas(c: &mut Criterion) {
    let mut group = c.benchmark_group("cublas");

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();

    // rng.gen::<u16>()

    let a_host = (0..DB_SIZE * WIDTH).map(|_| 1u16).collect::<Vec<_>>();
    let b_host = (0..QUERY_SIZE * WIDTH).map(|_| 1u16).collect::<Vec<_>>();

    let a1_host = a_host.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
    let a0_host = a_host.iter().map(|x| (x & 0xFF) as u8).collect::<Vec<_>>();
    let b1_host = b_host.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
    let b0_host = b_host.iter().map(|x| (x & 0xFF) as u8).collect::<Vec<_>>();

    let a1_dev = dev.htod_sync_copy(&a1_host).unwrap();
    let a0_dev = dev.htod_sync_copy(&a0_host).unwrap();
    let b1_dev = dev.htod_sync_copy(&b1_host).unwrap();
    let b0_dev = dev.htod_sync_copy(&b0_host).unwrap();

    let mut c_host = vec![0u32; DB_SIZE * QUERY_SIZE * 3];
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();
    
    group.bench_function(format!("cublas u16 mul with int8 {} x {}", DB_SIZE, QUERY_SIZE), |b| {
        b.iter(|| {
            gemm(&blas.handle(), &a0_dev, &b0_dev, &mut c_dev, 0);
            gemm(&blas.handle(), &a0_dev, &b1_dev, &mut c_dev, (DB_SIZE * QUERY_SIZE * 4 * 1) as u64);
            gemm(&blas.handle(), &a1_dev, &b0_dev, &mut c_dev, (DB_SIZE * QUERY_SIZE * 4 * 2) as u64);

            dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();

            let c1 = &c_host[DB_SIZE * QUERY_SIZE * 0..DB_SIZE * QUERY_SIZE * 1];
            let c2 = &c_host[DB_SIZE * QUERY_SIZE * 1..DB_SIZE * QUERY_SIZE * 2];
            let c3 = &c_host[DB_SIZE * QUERY_SIZE * 2..DB_SIZE * QUERY_SIZE * 3];
            
            let res = c1
                .into_iter()
                .zip(c2)
                .zip(c3)
                .map(|((c1, c2), c3)| (c1 + ((c2 + c3) << 8)) as u16)
                .collect::<Vec<_>>();

            // let res = c1.into_par_iter() 
            //             .zip(c2.into_par_iter())
            //             .zip(c3.into_par_iter())
            //             .zip(c4.into_par_iter())
            //             .map(|(((c1, c2), c3), c4)| (c4 + ((c3 + c2) << 8) + (c1 << 16)) as u16)
            //             .collect::<Vec<_>>();

            black_box(res);
        });
        // assert!(res.iter().all(|x| *x == 12800));
    });
    
    // Have to debug this, unable to get this make sense
    // Vanilla ndArray version for sanity check (have to use column major)
    // let a_nda = Array2::from_shape_vec(
    //     (WIDTH as usize, DB_SIZE as usize),
    //     a_host.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let b_nda = Array2::from_shape_vec(
    //     (WIDTH as usize, QUERY_SIZE as usize),
    //     b_host.into_iter().map(|x| x as u32).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let c_nda = a_nda.t().dot(&b_nda).into_raw_vec();

    // assert_eq!(
    //     c_nda[0..100],
    //     res[0..100],
    //     "GPU result does not match CPU implementation"
    // );

    group.finish();
}

criterion_group!(benches, cublas,);
criterion_main!(benches);
