use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gpu_bench::{ComputeDataType, MatmulEngineU16, MatmulEngineU32};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WIDTH: usize = 12_800;
const DB_SIZE: usize = 100_000;
const CHUNK_SIZE: usize = 25_000;
const RNG_SEED: u64 = 40;
const QUERY_SIZES: &[usize] = &[31, 155, 310, 620, 930, 1550, 2170];

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

        let mut gpu_result = vec![0u16; DB_SIZE * query_size];

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, query_size, ComputeDataType::U16, None);

        let preprocessed_query = engine.preprocess_query(&query);

        group.bench_function(
            format!("u16 x u16 → u16 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
                });
            },
        );
    }
}

fn bench_p16(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_p16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    const P: u16 = 65529; // (1 << 16) - 7

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let mut gpu_result = vec![0u16; DB_SIZE * query_size];

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, query_size, ComputeDataType::P16, Some(P));

        let preprocessed_query = engine.preprocess_query(&query);

        group.bench_function(
            format!("p16 x p16 → p16 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
                });
            },
        );
    }
}

fn bench_u16u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u16u32");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen::<u16>())
            .collect::<Vec<_>>();

        let mut gpu_result = vec![0u32; DB_SIZE * query_size];

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU16::<u32>::create(&db, WIDTH, query_size, ComputeDataType::U32, None);

        let preprocessed_query = engine.preprocess_query(&query);

        group.bench_function(
            format!("u16 x u16 → u32 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
                });
            },
        );
    }
}

fn bench_p14(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_p14");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    const P: u16 = (1 << 14) - 3;

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen_range(0..P))
            .collect::<Vec<_>>();

        let mut gpu_result = vec![0u16; DB_SIZE * query_size];

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, query_size, ComputeDataType::P14, Some(P));
        let preprocessed_query = engine.preprocess_query(&query);

        group.bench_function(
            format!("p14 x p14 → p14 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
                });
            },
        );
    }
}

fn bench_u14(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u14");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for &query_size in QUERY_SIZES {
        let db = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..(1 << 14)))
            .collect::<Vec<_>>();

        let query = (0..query_size * WIDTH)
            .map(|_| rng.gen_range(0..(1 << 14)))
            .collect::<Vec<_>>();

        let mut gpu_result = vec![0u16; DB_SIZE * query_size];
        // unsafe {
        //     cuMemAllocHost_v2(gpu_result.as_mut_ptr() as *mut _, DB_SIZE * query_size);
        // }

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU16::<u16>::create(&db, WIDTH, query_size, ComputeDataType::U14, None);
        let preprocessed_query = engine.preprocess_query(&query);
        group.bench_function(
            format!("u14 x u14 → u14 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
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

        let mut gpu_result = vec![0u32; DB_SIZE * query_size];
        // unsafe {
        //     cuMemAllocHost_v2(gpu_result.as_mut_ptr() as *mut _, DB_SIZE * query_size);
        // }

        group.throughput(Throughput::Elements((DB_SIZE * query_size / 31) as u64));
        let mut engine =
            MatmulEngineU32::create(&db, WIDTH, query_size, CHUNK_SIZE);
        let preprocessed_query = engine.preprocess_query(&query);

        group.bench_function(
            format!("u32 x u32 → u32 ({} x {})", DB_SIZE, query_size),
            |b| {
                b.iter(|| {
                    black_box(engine.dot(&preprocessed_query, &mut gpu_result));
                });
            },
        );
    }
}

// criterion_group!(benches, bench_u16, bench_p16, bench_u16u32, bench_p14, bench_u14, bench_u32);
criterion_group!(benches, bench_u32);
criterion_main!(benches);
