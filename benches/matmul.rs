use std::ffi::c_void;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use cudarc::driver::sys::cuMemAllocHost_v2;
use gpu_bench::{matmul::MatmulEngine, ComputeDataType, MatmulEngineU16, MatmulEngineU32};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WIDTH: usize = 12_800;
const DB_SIZE: usize = 300_000;
const CHUNK_SIZES: &[usize] = &[50_000];
const RNG_SEED: u64 = 40;
const QUERY_SIZES: &[usize] = &[31, 62, 93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465, 496, 527, 558, 589, 620, 651, 682, 713, 744, 775, 806, 837, 868, 899, 930, 961, 992, 1023, 1054, 1085, 1116, 1147, 1178, 1209, 1240, 1271, 1302, 1333, 1364, 1395, 1426, 1457, 1488, 1519, 1550, 1581, 1612, 1643, 1674, 1705, 1736, 1767, 1798, 1829, 1860, 1891, 1922, 1953, 1984, 2015, 2046, 2077, 2108, 2139, 2170, 2201, 2232, 2263, 2294, 2325, 2356, 2387, 2418, 2449, 2480, 2511, 2542, 2573, 2604, 2635, 2666, 2697, 2728, 2759, 2790, 2821, 2852, 2883, 2914, 2945, 2976, 3007, 3038, 3069];

fn bench_u16(c: &mut Criterion) {
    for &chunk_size in CHUNK_SIZES {
        let mut group = c.benchmark_group(format!("bench_u16_{}", chunk_size));
        group.sample_size(10);
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for &query_size in QUERY_SIZES {
            let db = (0..DB_SIZE * WIDTH)
                .map(|_| rng.gen::<u16>())
                .collect::<Vec<_>>();

            let query = (0..query_size * WIDTH)
                .map(|_| rng.gen::<u16>())
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
            let mut engine = MatmulEngine::create(&db, WIDTH, query_size, chunk_size, None);
            let preprocessed_query = engine.preprocess_query(&query);
            let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * query_size * 2);
            }

            group.bench_function(
                format!("u16 x u16 → u16 ({} x {})", DB_SIZE, query_size),
                |b| {
                    b.iter(|| {
                        black_box(engine.dot(&preprocessed_query, results_host_ptr as *mut u16));
                    });
                },
            );
        }
    }
}

fn bench_p16(c: &mut Criterion) {
    const P: u16 = ((1u32 << 16) - 17) as u16;
    for &chunk_size in CHUNK_SIZES {
        let mut group = c.benchmark_group(format!("bench_p16_{}", chunk_size));
        group.sample_size(10);
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for &query_size in QUERY_SIZES {
            let db = (0..DB_SIZE * WIDTH)
                .map(|_| rng.gen_range(0..P))
                .collect::<Vec<_>>();

            let query = (0..query_size * WIDTH)
                .map(|_| rng.gen_range(0..P))
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
            let mut engine = MatmulEngine::create(&db, WIDTH, query_size, chunk_size, Some(P));
            let preprocessed_query = engine.preprocess_query(&query);
            let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * query_size * 2);
            }

            group.bench_function(
                format!("p16 x p16 → p16 ({} x {})", DB_SIZE, query_size),
                |b| {
                    b.iter(|| {
                        black_box(engine.dot(&preprocessed_query, results_host_ptr as *mut u16));
                    });
                },
            );
        }
    }
}

fn bench_u32(c: &mut Criterion) {
    for &chunk_size in CHUNK_SIZES {
        let mut group = c.benchmark_group(format!("bench_u32_{}", chunk_size));
        group.sample_size(10);
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for &query_size in QUERY_SIZES {
            let db = (0..DB_SIZE * WIDTH)
                .map(|_| rng.gen::<u32>())
                .collect::<Vec<_>>();

            let query = (0..query_size * WIDTH)
                .map(|_| rng.gen::<u32>())
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
            let mut engine = MatmulEngine::create(&db, WIDTH, query_size, chunk_size, None);
            let preprocessed_query = engine.preprocess_query(&query);
            let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * query_size * 4);
            }

            group.bench_function(
                format!("u32 x u32 → u32 ({} x {})", DB_SIZE, query_size),
                |b| {
                    b.iter(|| {
                        black_box(engine.dot(&preprocessed_query, results_host_ptr as *mut u32));
                    });
                },
            );
        }
    }
}

fn bench_p32(c: &mut Criterion) {
    const P: u32 = 4294967291;
    for &chunk_size in CHUNK_SIZES {
        let mut group = c.benchmark_group(format!("bench_p32_{}", chunk_size));
        group.sample_size(10);
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for &query_size in QUERY_SIZES {
            let db = (0..DB_SIZE * WIDTH)
                .map(|_| rng.gen_range(0..P))
                .collect::<Vec<_>>();

            let query = (0..query_size * WIDTH)
                .map(|_| rng.gen_range(0..P))
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
            let mut engine = MatmulEngine::create(&db, WIDTH, query_size, chunk_size, Some(P));
            let preprocessed_query = engine.preprocess_query(&query);
            let mut results_host_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = cuMemAllocHost_v2(&mut results_host_ptr, DB_SIZE * query_size * 4);
            }

            group.bench_function(
                format!("p32 x p32 → p32 ({} x {})", DB_SIZE, query_size),
                |b| {
                    b.iter(|| {
                        black_box(engine.dot(&preprocessed_query, results_host_ptr as *mut u32));
                    });
                },
            );
        }
    }
}

criterion_group!(benches, bench_u16, bench_p16, bench_u32, bench_p32);
criterion_main!(benches);
