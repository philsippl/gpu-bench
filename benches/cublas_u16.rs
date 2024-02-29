use std::ffi::c_void;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cudarc::cublas::result::gemm_ex;
use cudarc::cublas::{sys, CudaBlas};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 930;
const DB_SIZE: usize = 100_000;
const RNG_SEED: u64 = 42;

const PTX_SRC: &str = "
extern \"C\" __global__ void calc_u16(int* c, unsigned short* output, unsigned short* a0Sums, unsigned short* a1Sums, unsigned short* b0Sums, unsigned short* b1Sums, size_t numRows, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Correct the results to simulate u8
        unsigned short c00 = c[idx];
        c00 += (a0Sums[idx % numRows] + b0Sums[idx / numRows]) << 7;
        unsigned short c01 = c[idx + numElements];
        c01 += (a0Sums[idx % numRows] + b1Sums[idx / numRows]) << 7;
        unsigned short c10 = c[idx + numElements * 2];
        c10 += (a1Sums[idx % numRows] + b0Sums[idx / numRows]) << 7;

        // Calculate the u16 result
        output[idx] = c00 + ((c01 + c10) << 8);
    }
}
";

fn gemm(
    handle: &sys::cublasHandle_t,
    a: &CudaSlice<i8>,
    b: &CudaSlice<i8>,
    c: &mut CudaSlice<i32>,
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

fn calculate_sum(a: &Vec<u8>, size: usize) -> Vec<u16> {
    a.chunks(size)
        .map(|row| row.iter().map(|&x| x as u16).sum())
        .collect()
}

fn preprocess(a: &Vec<u8>) -> Vec<i8> {
    a.iter().map(|x| (*x as i8 - 127 - 1)).collect()
}

fn cublas(c: &mut Criterion) {
    let mut group = c.benchmark_group("cublas");

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();

    let a_host = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();
    let b_host = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();

    let a1_host = a_host.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
    let a1_sums = calculate_sum(&a1_host, WIDTH);
    let a1_host = preprocess(&a1_host);

    let a0_host = a_host.iter().map(|x| (x & 0xFF) as u8).collect::<Vec<_>>();
    let a0_sums = calculate_sum(&a0_host, WIDTH);
    let a0_host = preprocess(&a0_host);

    let b1_host = b_host.iter().map(|x| (x >> 8) as u8).collect::<Vec<_>>();
    let b0_host = b_host.iter().map(|x| (x & 0xFF) as u8).collect::<Vec<_>>();

    let a1_dev = dev.htod_sync_copy(&a1_host).unwrap();
    let a0_dev = dev.htod_sync_copy(&a0_host).unwrap();
    let a1_sums_dev = dev.htod_sync_copy(&a1_sums).unwrap();
    let a0_sums_dev = dev.htod_sync_copy(&a0_sums).unwrap();

    let c_host = vec![0i32; DB_SIZE * QUERY_SIZE * 3];
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();
    let mut final_host = vec![0u16; DB_SIZE * QUERY_SIZE];
    let mut final_dev = dev.htod_sync_copy(&final_host).unwrap();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    dev.load_ptx(ptx, "calc_u16", &["calc_u16"]).unwrap();
    let f = dev.get_func("calc_u16", "calc_u16").unwrap();

    let num_elements = DB_SIZE * QUERY_SIZE;
    let threads_per_block = 256;
    let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        block_dim: (threads_per_block as u32, 1, 1),
        grid_dim: (blocks_per_grid as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // TODO: improve
    let b1_sums = calculate_sum(&b1_host, WIDTH);
    let b1_host = preprocess(&b1_host);
    let b0_sums = calculate_sum(&b0_host, WIDTH);
    let b0_host = preprocess(&b0_host);

    group.bench_function(
        format!("cublas u16 mul with int8 {} x {}", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                let b1_dev = dev.htod_sync_copy(&b1_host).unwrap();
                let b0_dev = dev.htod_sync_copy(&b0_host).unwrap();
                let b1_sums_dev = dev.htod_sync_copy(&b1_sums).unwrap();
                let b0_sums_dev = dev.htod_sync_copy(&b0_sums).unwrap();

                gemm(&blas.handle(), &a0_dev, &b0_dev, &mut c_dev, 0);
                gemm(
                    &blas.handle(),
                    &a0_dev,
                    &b1_dev,
                    &mut c_dev,
                    (DB_SIZE * QUERY_SIZE * 4 * 1) as u64,
                );
                gemm(
                    &blas.handle(),
                    &a1_dev,
                    &b0_dev,
                    &mut c_dev,
                    (DB_SIZE * QUERY_SIZE * 4 * 2) as u64,
                );

                unsafe {
                    f.clone().launch(
                        cfg,
                        (
                            &c_dev,
                            &mut final_dev,
                            &a0_sums_dev,
                            &a1_sums_dev,
                            &b0_sums_dev,
                            &b1_sums_dev,
                            DB_SIZE as u64,
                            (DB_SIZE * QUERY_SIZE) as u64,
                        ),
                    )
                }
                .unwrap();

                dev.dtoh_sync_copy_into(&final_dev, &mut final_host)
                    .unwrap();
            });
        },
    );

    // let a_nda = Array2::from_shape_vec(
    //     (DB_SIZE as usize, WIDTH as usize),
    //     a_host.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let b_nda = Array2::from_shape_vec(
    //     (QUERY_SIZE as usize, WIDTH as usize),
    //     b_host.into_iter().map(|x| x as u16).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let c_nda = a_nda.dot(&b_nda.t());

    // let mut vec_column_major: Vec<u16> = Vec::new();
    // for col in 0..c_nda.ncols() {
    //     for row in c_nda.column(col) {
    //         vec_column_major.push(*row);
    //     }
    // }

    // assert_eq!(
    //     vec_column_major,
    //     final_host,
    //     "GPU result does not match CPU implementation"
    // );

    group.finish();
}

criterion_group!(benches, cublas,);
criterion_main!(benches);
