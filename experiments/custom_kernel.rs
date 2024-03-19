/// Custom tiling kernel (each thread calculates one output element)
use criterion::{criterion_group, criterion_main, Criterion};

use cudarc::driver::sys::cuMemAllocHost_v2;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: u32 = 12_800;
const QUERY_SIZE: u32 = 32;
const DB_SIZE: u32 = 100_000;
const TILE_WIDTH: u32 = 32;
const RNG_SEED: u64 = 42;

const PTX_SRC: &str = "
#define TILE_WIDTH 32

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
        C[Row * N + Col] = tmpSum;
}
";

fn custom_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_kernel");

    let dev = CudaDevice::new(0).unwrap();
    let ptx = compile_ptx(PTX_SRC).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    dev.load_ptx(ptx, "matmul", &["matmul"]).unwrap();
    let f = dev.get_func("matmul", "matmul").unwrap();

    let a_host = (0..DB_SIZE * WIDTH).map(|_| rng.gen::<u16>() as u16).collect::<Vec<_>>();
    let b_host = (0..QUERY_SIZE * WIDTH).map(|_| rng.gen_range(0..2) as u16).collect::<Vec<_>>();
    
    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();

    let cfg = LaunchConfig {
        block_dim: (TILE_WIDTH, TILE_WIDTH, 1),
        grid_dim: (
            QUERY_SIZE.div_ceil(TILE_WIDTH),
            DB_SIZE.div_ceil(TILE_WIDTH),
            1,
        ),
        shared_mem_bytes: 0,
    };

    let mut c_host = vec![0u16; (DB_SIZE * QUERY_SIZE) as usize];
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

    unsafe {
        cuMemAllocHost_v2(c_host.as_mut_ptr() as *mut _, (DB_SIZE * QUERY_SIZE* 2) as usize);
    }

    group.bench_function(
        format!("custom kernel naïve {} x {}", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                unsafe {
                    f.clone().launch(
                        cfg,
                        (
                            &a_dev,
                            &b_dev,
                            &mut c_dev,
                            DB_SIZE as i32,
                            QUERY_SIZE as i32,
                            WIDTH as i32,
                        ),
                    )
                }
                .unwrap();

                dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();
            });
        },
    );

    // Vanilla ndArray version for sanity check
    let a_nda = Array2::from_shape_vec((DB_SIZE as usize, WIDTH as usize), a_host.clone()).unwrap();
    let b_nda =
        Array2::from_shape_vec((WIDTH as usize, QUERY_SIZE as usize), b_host.clone()).unwrap();
    let c_nda = a_nda.dot(&b_nda).into_raw_vec();
    assert_eq!(c_nda, c_host, "GPU result does not match CPU impl");

    group.finish();
}

criterion_group!(benches, custom_kernel,);
criterion_main!(benches);
