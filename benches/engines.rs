use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gpu_bench::{ComputeDataType, MatmulEngine};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WIDTH: usize = 12_800;
const QUERY_SIZE: usize = 31;
const DB_SIZE: usize = 1000;
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
    let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::U16, None);

    group.bench_function(
        format!("u16 x u16 → u16 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot(&query));
            });
        },
    );
}

fn bench_p16(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_p16");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    const P: u16 = 65529;

    let db = (0..DB_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    let query = (0..QUERY_SIZE * WIDTH)
        .map(|_| rng.gen_range(0..P))
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    let mut engine = MatmulEngine::create(&db, WIDTH, QUERY_SIZE, ComputeDataType::P16, Some(P));

    group.bench_function(
        format!("p16 x p16 → p16 ({} x {})", DB_SIZE, QUERY_SIZE),
        |b| {
            b.iter(|| {
                black_box(engine.dot(&query));
            });
        },
    );
}

criterion_group!(benches, bench_u16, bench_p16);
criterion_main!(benches);
