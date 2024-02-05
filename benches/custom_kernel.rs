use std::os::raw::c_void;

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
#define TILE_WIDTH 32  // Define the width of the tile

extern \"C\" __global__ void matmul(unsigned short* A, unsigned short* B, unsigned short* C, int M, int N, int K) {
    __shared__ unsigned short As[TILE_WIDTH][TILE_WIDTH];
    __shared__ unsigned short Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    unsigned short tmpSum = 0;

    for (int t = 0; t < (K - 1) / TILE_WIDTH + 1; ++t) {
        if (Row < M && t * TILE_WIDTH + tx < K)
            As[ty][tx] = A[Row * K + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0;

        if (Col < N && t * TILE_WIDTH + ty < K)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + Col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            tmpSum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (Row < M && Col < N)
        C[Row * N + Col] += tmpSum;
}
";

fn create_random_matrix(n: usize, m: usize) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    (0..n * m).map(|_| rng.gen::<u16>()).collect()
}

fn custom_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_kernel");

    let dev = CudaDevice::new(0).unwrap();
    let ptx = compile_ptx(PTX_SRC).unwrap();
    dev.load_ptx(ptx, "matmul", &["matmul"]).unwrap();
    let f = dev.get_func("matmul", "matmul").unwrap();

    let a_host = create_random_matrix(DB_SIZE, WIDTH);
    let b_host = create_random_matrix(QUERY_SIZE, WIDTH);
    let mut c_host = vec![0f64; DB_SIZE * QUERY_SIZE];

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

    let cfg = LaunchConfig {
        block_dim: (TILE_WIDTH, TILE_WIDTH, 1),
        grid_dim: (
            (query_size as u32 + TILE_WIDTH - 1) / TILE_WIDTH,
            (db_size as u32 + TILE_WIDTH - 1) / TILE_WIDTH,
            1,
        ),
        shared_mem_bytes: 0,
    };

    group.bench_function("matmul u16", |b| {
        b.iter(|| {
            unsafe {
                f.launch(
                    cfg,
                    (
                        &a_dev,
                        &b_dev,
                        &mut c_dev,
                        db_size as i32,
                        query_size as i32,
                        width as i32,
                    ),
                )
            }
            .unwrap();

            dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();
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

criterion_group!(benches, custom_kernel,);
criterion_main!(benches);
