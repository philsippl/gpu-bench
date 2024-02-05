use cudarc::cublas::result::dgemm;
use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys::{cuMemAllocHost_v2, cuMemcpyDtoH_v2};
use cudarc::driver::{CudaDevice, DevicePtr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TILE_WIDTH: u32 = 32;
const WIDTH: usize = 12800;
const DB_SIZE: usize = 100000;
const QUERY_SIZE: usize = 31;
const RNG_SEED: u64 = 42;

const PTX_SRC: &str = "
extern \"C\" __global__ void carsten(double* input, unsigned short* output, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = input[idx];
    }
}
";

fn create_random_matrix(n: usize, m: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let total_size = n * m;
    (0..total_size).map(|_| rng.gen::<u16>() as f64).collect()
}

fn cublas(c: &mut Criterion) {
    let mut group = c.benchmark_group("cublas");

    let dev = CudaDevice::new(0).unwrap();
    let stream = dev.fork_default_stream().unwrap();
    let ptx = compile_ptx(PTX_SRC).unwrap();
    dev.load_ptx(ptx, "carsten", &["carsten"]).unwrap();
    let f = dev.get_func("carsten", "carsten").unwrap();

    let a_host = create_random_matrix(DB_SIZE, WIDTH);
    let b_host = create_random_matrix(QUERY_SIZE, WIDTH);
    let mut c_host = vec![0f64; DB_SIZE * QUERY_SIZE];
    let mut final_host = vec![0u16; DB_SIZE * QUERY_SIZE];

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();
    let mut final_dev = dev.htod_sync_copy(&final_host).unwrap();

    let blas = CudaBlas::new(dev.clone()).unwrap();
    unsafe {
        blas.set_stream(Some(&stream));
    }

    let num_elements = DB_SIZE * QUERY_SIZE;
    let threads_per_block = 256;
    let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        block_dim: (threads_per_block as u32, 1, 1),
        grid_dim: (blocks_per_grid as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    group.bench_function("matmul u16", |b| {
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
            }
            .unwrap();

            // Launch the cast kernel
            unsafe {
                f.launch_on_stream(
                    &stream,
                    cfg,
                    (&c_dev, &mut final_dev, (DB_SIZE * QUERY_SIZE) as i32),
                )
            }
            .unwrap();

            dev.wait_for(&stream).unwrap();

            dev.dtoh_sync_copy_into(&final_dev, &mut final_host)
                .unwrap();
        });
    });

    group.finish();

    // unsafe {
    //     cuMemAllocHost_v2(c_host.as_mut_ptr() as *mut _, db_size * query_size*2);
    // }

    // let mut c_host_ptr: *mut c_void = std::ptr::null_mut();
    // let bytesize = db_size * query_size * std::mem::size_of::<u16>();
    // unsafe {
    //     let _ = cuMemAllocHost_v2(&mut c_host_ptr, bytesize);
    // }
    //
    // unsafe {
    //     let _ = cuMemcpyDtoH_v2(c_host_ptr, *c_dev.device_ptr(), bytesize);
    // }
}
