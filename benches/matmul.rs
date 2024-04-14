use std::ffi::c_void;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use cudarc::driver::sys::cuMemAllocHost_v2;
use gpu_bench::{matmul::MatmulEngine, ComputeDataType, MatmulEngineU16, MatmulEngineU32};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WIDTH: usize = 12_800;
const DB_SIZE: usize = 300_000;
const CHUNK_SIZE: usize = 10_000;
const RNG_SEED: u64 = 40;
const QUERY_SIZES: &[usize] = &[310, 620, 930, 1550, 2480];

fn bench_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine = MatmulEngine::create(&db, WIDTH, query_size, CHUNK_SIZE, None);
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

fn bench_p16(c: &mut Criterion) {
    const P: u16 = ((1u32 << 16) - 17) as u16;
    let mut group = c.benchmark_group("bench_p16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine = MatmulEngine::create(&db, WIDTH, query_size, CHUNK_SIZE, Some(P));
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

fn bench_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u32");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u32>())
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen::<u32>())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine = MatmulEngine::create(&db, WIDTH, query_size, CHUNK_SIZE, None);
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

fn bench_p32(c: &mut Criterion) {
    const P: u32 = 4294967291;
    let mut group = c.benchmark_group("bench_p32");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine = MatmulEngine::create(&db, WIDTH, query_size, CHUNK_SIZE, Some(P));
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

criterion_group!(benches, bench_u16, bench_p16, bench_u32, bench_p32);
criterion_main!(benches);
