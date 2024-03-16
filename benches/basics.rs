use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaSlice};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

const WIDTH: usize = 12_800;
const RNG_SEED: u64 = 40;

fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_decomposition");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for query_size in [1, 5, 10, 20, 30, 40, 5, 100] {
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

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();

    for query_size in [1, 5, 10, 20, 30, 40, 5, 100] {
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

fn bench_rowsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_rowsum");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dev = CudaDevice::new(0).unwrap();
    let blas = CudaBlas::new(dev.clone()).unwrap();

    for query_size in [1, 10, 30, 50, 100, 1000, 10000] {
        let query = (0..query_size * 31 * WIDTH)
            .map(|_| rng.gen::<u8>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((query_size / 31) as u64));
        // group.bench_function(format!("rowsum CPU ({} x {})", query_size, WIDTH), |b| {
        //     b.iter(|| {
        //         let sums: Vec<u32> = query
        //             .par_chunks(WIDTH)
        //             .map(|row| row.iter().map(|&x| x as u32).sum())
        //             .collect();
        //         black_box(sums);
        //     });
        // });

        let b1_dev = dev.htod_sync_copy(&query).unwrap();
        let mut query1_sums: CudaSlice<i32> = dev.alloc_zeros(query_size).unwrap();
        let ones = vec![1u8; WIDTH];
        let ones = dev.htod_sync_copy(&ones).unwrap();
        let mut results = vec![1i32; query_size];

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
                dev
                    .dtoh_sync_copy_into(&query1_sums, &mut results)
                    .unwrap();
            });
        });
    }
}

criterion_group!(benches, bench_rowsum, bench_memcpy, bench_decomposition);
criterion_main!(benches);
