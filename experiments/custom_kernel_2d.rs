/// Custom 2D tiling kernel
/// From: https://siboehm.com/articles/22/CUDA-MMM
/// Sanity check fails, need to investigate
use criterion::{criterion_group, criterion_main, Criterion};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: u32 = 12_800;
const DB_SIZE: u32 = 100_000;
const QUERY_SIZE: u32 = 32;
const RNG_SEED: u64 = 42;
const BM: u32 = 32;
const BN: u32 = 32;
const TM: u32 = 4;
const TN: u32 = 4;

const PTX_SRC: &str = "
#define BM 32
#define BN 32
#define BK 4
#define TM 4
#define TN 4

extern \"C\" __global__ void matmul(unsigned short* A, unsigned short* B, unsigned short* C, int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const int strideA = numThreadsBlocktile / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const int strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
    unsigned short threadResults[TM * TN] = {0};
    // register caches for As and Bs
    unsigned short regM[TM] = {0};
    unsigned short regN[TN] = {0};

    // outer-most loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
        C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            threadResults[resIdxM * TN + resIdxN];
        }
    }
}
";

fn custom_kernel_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_kernel");

    let dev = CudaDevice::new(0).unwrap();
    let ptx = compile_ptx(PTX_SRC).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    dev.load_ptx(ptx, "matmul", &["matmul"]).unwrap();
    let f = dev.get_func("matmul", "matmul").unwrap();

    let a_host = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>() as u16)
        .collect::<Vec<_>>();
    let b_host = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..2) as u16)
        .collect::<Vec<_>>();
    let mut c_host = vec![0u16; (DB_SIZE * QUERY_SIZE) as usize];

    // unsafe {
    //     cuMemAllocHost_v2(c_host.as_mut_ptr() as *mut _, DB_SIZE * QUERY_SIZE * 2);
    // }

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

    let cfg = LaunchConfig {
        block_dim: ((BM * BN) / (TM * TN), 1, 1),
        grid_dim: (QUERY_SIZE.div_ceil(BN), DB_SIZE.div_ceil(BM), 1),
        shared_mem_bytes: 8192,
    };

    group.bench_function(
        format!("custom kernel 2d {} x {}", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                unsafe {
                    f.clone().launch(
                        cfg,
                        (
                            &a_dev,
                            &b_dev,
                            &mut c_dev,
                            DB_SIZE as u64,
                            QUERY_SIZE as u64,
                            WIDTH as u64,
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

criterion_group!(benches, custom_kernel_2d,);
criterion_main!(benches);
