use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use cudarc::driver::CudaDevice;
use gpu_bench::{ComputeDataType, MatmulEngine};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 310;
const DB_SIZE: usize = 200_000;
const RNG_SEED: u64 = 40;

fn bench_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine =
        MatmulEngine::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U16, None);

    engine.preprocess_query(&query);

    group.bench_function(
        format!("u16 x u16 → u16 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot());
            });
        },
    );
}

fn bench_p16(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_p16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    const P: u16 = 65529; // (1 << 16) - 7

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine =
        MatmulEngine::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::P16, Some(P));

    engine.preprocess_query(&query);

    group.bench_function(
        format!("p16 x p16 → p16 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot());
            });
        },
    );
}

fn bench_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u32");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen::<u16>())
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine =
        MatmulEngine::<u32>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U32, None);

    engine.preprocess_query(&query);

    group.bench_function(
        format!("u32 x u32 → u32 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot());
            });
        },
    );
}

fn bench_p14(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_p14");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    const P: u16 = (1 << 14) - 3;

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine =
        MatmulEngine::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::P14, Some(P));

    engine.preprocess_query(&query);

    group.bench_function(
        format!("p14 x p14 → p14 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot());
            });
        },
    );
}

fn bench_u14(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_u14");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..(1 << 14)))
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..(1 << 14)))
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine =
        MatmulEngine::<u16>::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U14, None);

    engine.preprocess_query(&query);

    group.bench_function(
        format!("u14 x u14 → u14 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot());
            });
        },
    );
}

criterion_group!(benches, bench_u16, bench_p16, bench_u32, bench_p14, bench_u14);
// criterion_group!(benches, bench_p14);
criterion_main!(benches);
