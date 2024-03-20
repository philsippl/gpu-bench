use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use cudarc::cublas::CudaBlas;
use cudarc::driver::sys::cuMemAllocHost_v2;
use cudarc::driver::{CudaDevice, CudaSlice};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

const WIDTH: usize = 12_800;
const RNG_SEED: u64 = 40;

fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_decomposition");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for query_size in [1, 5, 10, 20, 30, 40, 50, 100] {
        let query = (0..query_size * 31 * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(query_size as u64));
        group.bench_function(
            format!("u16 decomposition threaded ({} x {})", query_size, WIDTH),
            |b| {
                b.iter(|| {
                    let (a, b) = query
                        .par_iter()
                        .map(|&x| ((x >> 7) as u8, (x & 0x7F) as u8))
                        .collect::<(Vec<_>, Vec<_>)>();
                    black_box((a, b));
                });
            },
        );

        group.bench_function(
            format!("u16 decomposition ({} x {})", query_size, WIDTH),
            |b| {
                b.iter(|| {
                    let (a, b): (Vec<u8>, Vec<u8>) = query
                        .iter()
                        .map(|&x| ((x >> 7) as u8, (x & 0x7F) as u8))
                        .unzip();
                    black_box((a, b));
                });
            },
        );
    }
}

fn bench_memcpy_htod(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();

    for query_size in [1, 5, 10, 30, 50, 100, 1000] {
        let query = (0..query_size * 31 * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Bytes((query_size * 31 * WIDTH * 2) as u64));
        group.bench_function(
            format!("host to device memcpy ({} x {})", query_size, WIDTH),
            |b| {
                b.iter(|| {
                    black_box(dev.htod_sync_copy(&query).unwrap());
                });
            },
        );
    }
}

fn bench_memcpy_dtoh(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");
    let dev = CudaDevice::new(0).unwrap();
    const DB_SIZE: usize = 200_000;

    for query_size in [930] {
        let data: CudaSlice<u8> = dev.alloc_zeros(query_size * DB_SIZE * 2).unwrap();
        let mut result = vec![0u8; query_size * DB_SIZE * 2];

        unsafe {
            cuMemAllocHost_v2(result.as_mut_ptr() as *mut _, query_size * DB_SIZE * 2);
        }

        group.throughput(Throughput::Bytes((query_size * DB_SIZE * 2) as u64));
        group.bench_function(
            format!("device to host memcpy ({} x {})", query_size, DB_SIZE),
            |b| {
                b.iter(|| {
                    black_box(dev.dtoh_sync_copy_into(&data, &mut result).unwrap());
                });
            },
        );
    }
}

fn bench_rowsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_rowsum");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();

    for query_size in [50, 100, 1000] {
        let query = (0..query_size * 31 * WIDTH)
            .map(|_| rng.gen::<u8>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((query_size / 31) as u64));
        group.bench_function(format!("rowsum CPU ({} x {})", query_size, WIDTH), |b| {
            b.iter(|| {
                let sums: Vec<u32> = query
                    .par_chunks(WIDTH)
                    .map(|row| row.iter().map(|&x| x as u32).sum())
                    .collect();
                black_box(sums);
            });
        });

        let b1_dev = dev.htod_sync_copy(&query).unwrap();
        let mut query1_sums: CudaSlice<i32> = dev.alloc_zeros(query_size).unwrap();
        let ones = vec![1u8; WIDTH];
        let ones = dev.htod_sync_copy(&ones).unwrap();

        group.bench_function(format!("rowsum cuBLAS ({} x {})", query_size, WIDTH), |b| {
            b.iter(|| {
                gpu_bench::gemm(
                    &blas.handle(),
                    &b1_dev,
                    &ones,
                    &mut query1_sums,
                    0,
                    query_size,
                    1,
                    WIDTH,
                );

                // Make very sure this is not async
                dev.synchronize().unwrap();
            });
        });
    }
}

fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_gemm");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();
    const DB_SIZE: usize = 100_000;

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen::<u8>())
        .collect::<Vec<_>>();

    let db_dev = dev.htod_sync_copy(&db).unwrap();

    for query_size in [31, 32, 155, 310, 620, 930] {
        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen::<u8>())
            .collect::<Vec<_>>();
        let b1_dev = dev.htod_sync_copy(&query).unwrap();
        let mut result: CudaSlice<i32> = dev.alloc_zeros(DB_SIZE * query_size).unwrap();

        for tp in vec![
            Throughput::Elements((DB_SIZE * query_size * WIDTH * 2) as u64),
            Throughput::Bytes(((DB_SIZE + query_size) * WIDTH) as u64),
        ] {
            group.throughput(tp);

            group.bench_function(format!("gemm cuBLAS ({} x {})", DB_SIZE, query_size), |b| {
                b.iter(|| {
                    gpu_bench::gemm(
                        &blas.handle(),
                        &db_dev,
                        &b1_dev,
                        &mut result,
                        0,
                        DB_SIZE,
                        query_size,
                        WIDTH,
                    );

                    dev.synchronize().unwrap();
                });
            });
        }
    }
}

// criterion_group!(benches, bench_rowsum, bench_memcpy, bench_decomposition, bench_gemm);
criterion_group!(benches, bench_memcpy_dtoh, bench_gemm);
criterion_main!(benches);
