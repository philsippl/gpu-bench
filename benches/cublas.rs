/// cuBLAS implemention for int8, fp32, fp64
use std::ffi::c_void;

use criterion::{criterion_group, criterion_main, Criterion};

use cudarc::cublas::result::gemm_ex;
use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig};

use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 32;
const DB_SIZE: usize = 1000;
const RNG_SEED: u64 = 42;

fn cublas(c: &mut Criterion) {
    let mut group = c.benchmark_group("cublas");

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();

    // INT8
    // We can use int8 to calculate mask weights.
    // TODO: eval int4
    {
        let a_host = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as u8)
            .collect::<Vec<_>>();
        let b_host = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as u8)
            .collect::<Vec<_>>();
        let mut c_host = vec![0u32; DB_SIZE * QUERY_SIZE];

        let a_dev = dev.htod_sync_copy(&a_host).unwrap();
        let b_dev = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

        group.bench_function(format!("cublas int8 {} x {}", DB_SIZE, QUERY_SIZE), |b| {
            b.iter(|| {
                unsafe {
                    // See the supported types here: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
                    gemm_ex(
                        blas.handle().clone(),
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        DB_SIZE as i32,
                        QUERY_SIZE as i32,
                        WIDTH as i32,
                        &1 as *const i32 as *const c_void,
                        *a_dev.device_ptr() as *const _,
                        sys::cublasDataType_t::CUDA_R_8I,
                        WIDTH as i32,
                        *b_dev.device_ptr() as *const _,
                        sys::cublasDataType_t::CUDA_R_8I,
                        WIDTH as i32,
                        &0 as *const i32 as *const c_void,
                        *c_dev.device_ptr_mut() as *mut _,
                        sys::cublasDataType_t::CUDA_R_32I,
                        DB_SIZE as i32,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .unwrap();
                }
                dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();
            });
        });

        // Vanilla ndArray version for sanity check (have to use column major)
        let a_nda = Array2::from_shape_vec(
            (DB_SIZE as usize,  WIDTH as usize),
            a_host.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
        )
        .unwrap();
        let b_nda = Array2::from_shape_vec(
            (QUERY_SIZE as usize, WIDTH as usize),
            b_host.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
        )
        .unwrap();
        let c_nda = a_nda
            .dot(&b_nda.t());
            // .into_raw_vec()
            // .into_iter()
            // .map(|x| x as u32)
            // .collect::<Vec<_>>();

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push(*row);
            }
        }

        assert_eq!(
            vec_column_major.into_iter().map(|x| x as u32).collect::<Vec<_>>()[0..100],
            c_host[0..100],
            "GPU result does not match CPU implementation"
        );
    }

    // FP32
    // This will overflow during accumulation: 16 bit * 12800 =~ 30 bit
    {
        let a_host = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>() as f32)
            .collect::<Vec<_>>();
        let b_host = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as f32)
            .collect::<Vec<_>>();
        let mut c_host = vec![0f32; DB_SIZE * QUERY_SIZE];

        let a_dev = dev.htod_sync_copy(&a_host).unwrap();
        let b_dev = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

        group.bench_function(format!("cublas fp32 {} x {}", DB_SIZE, QUERY_SIZE), |b| {
            b.iter(|| {
                unsafe {
                    blas.gemm(
                        GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: DB_SIZE as i32,
                            n: QUERY_SIZE as i32,
                            k: WIDTH as i32,
                            alpha: 1.0,
                            lda: DB_SIZE as i32,
                            ldb: WIDTH as i32,
                            beta: 0.0,
                            ldc: DB_SIZE as i32,
                        },
                        &a_dev,
                        &b_dev,
                        &mut c_dev,
                    )
                    .unwrap();
                }
                dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();
            });
        });

        // Skip sanity check bc of overflows
    }

    // FP64
    {
        let a_host = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>() as f64)
            .collect::<Vec<_>>();
        let b_host = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as f64)
            .collect::<Vec<_>>();
        let mut c_host = vec![0f64; DB_SIZE * QUERY_SIZE];

        let a_dev = dev.htod_sync_copy(&a_host).unwrap();
        let b_dev = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();
        group.bench_function(format!("cublas fp64 {} x {}", DB_SIZE, QUERY_SIZE), |b| {
            b.iter(|| {
                unsafe {
                    blas.gemm(
                        GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: DB_SIZE as i32,
                            n: QUERY_SIZE as i32,
                            k: WIDTH as i32,
                            alpha: 1.0,
                            lda: DB_SIZE as i32,
                            ldb: WIDTH as i32,
                            beta: 0.0,
                            ldc: DB_SIZE as i32,
                        },
                        &a_dev,
                        &b_dev,
                        &mut c_dev,
                    )
                    .unwrap();
                }
                dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, cublas,);
criterion_main!(benches);
