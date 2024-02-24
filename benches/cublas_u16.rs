use std::ffi::c_void;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cudarc::cublas::result::gemm_ex;
use cudarc::cublas::{sys, CudaBlas};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 320;
const DB_SIZE: usize = 100_000;
const RNG_SEED: u64 = 42;

const PTX_SRC: &str = "
extern \"C\" __global__ void calc_u16(int* c, short* output, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = (c[idx] + ((c[idx + numElements] + c[idx + numElements * 2]) << 8));
    }
}
";

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
    
    group.bench_function(format!("cublas u16 mul with int8 {} x {}", DB_SIZE, QUERY_SIZE), |b| {
        b.iter(|| {
            gemm(&blas.handle(), &a0_dev, &b0_dev, &mut c_dev, 0);
            gemm(&blas.handle(), &a0_dev, &b1_dev, &mut c_dev, (DB_SIZE * QUERY_SIZE * 4 * 1) as u64);
            gemm(&blas.handle(), &a1_dev, &b0_dev, &mut c_dev, (DB_SIZE * QUERY_SIZE * 4 * 2) as u64);

            unsafe {
                f.clone().launch(
                    cfg,
                    (&c_dev, &mut final_dev, (DB_SIZE * QUERY_SIZE) as u64),
                )
            }
            .unwrap();

            dev.dtoh_sync_copy_into(&final_dev, &mut final_host).unwrap();
        });

        // check
        assert!(final_host.iter().all(|x| *x == 12800));
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
