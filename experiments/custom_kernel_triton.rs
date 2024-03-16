use criterion::{criterion_group, criterion_main, Criterion};
use cudarc::driver::sys::cuMemAllocHost_v2;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WIDTH: usize = 12_800;
const DB_SIZE: usize = 100_000;
const RNG_SEED: u64 = 42;

fn custom_kernel_triton(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_kernel");

    let dev = CudaDevice::new(0).unwrap();
    let ptx = Ptx::from_file("matmul_kernel.ptx");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    dev.load_ptx(
        ptx,
        "matmul_kernel_0d1d2d3d4d5d",
        &["matmul_kernel_0d1d2d3d4d5d"],
    )
    .unwrap();
    let f = dev
        .get_func("matmul_kernel_0d1d2d3d4d5d", "matmul_kernel_0d1d2d3d4d5d")
        .unwrap();

    // Batchsize = 1
    {
        const QUERY_SIZE: usize = 32;
        let a_host = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>() as u16)
            .collect::<Vec<_>>();
        let b_host = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as u16)
            .collect::<Vec<_>>();
        let mut c_host = vec![0u16; (DB_SIZE * QUERY_SIZE) as usize];

        // Pin memory if DMA is available
        unsafe {
            cuMemAllocHost_v2(c_host.as_mut_ptr() as *mut _, DB_SIZE * QUERY_SIZE * 2);
        }

        let a_dev = dev.htod_sync_copy(&a_host).unwrap();
        let b_dev = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

        let cfg = LaunchConfig {
            block_dim: (64, 1, 1), // num_warps = 2 (warp = 32 threads)
            grid_dim: (
                (DB_SIZE.div_ceil(32) * QUERY_SIZE.div_ceil(32)) as u32,
                1,
                1,
            ),
            shared_mem_bytes: 8192,
        };

        group.bench_function(
            format!(
                "custom kernel triton {} x {} batchsize 1",
                DB_SIZE, QUERY_SIZE
            ),
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
        let a_nda = Array2::from_shape_vec((DB_SIZE, WIDTH), a_host.clone()).unwrap();
        let b_nda = Array2::from_shape_vec((WIDTH, QUERY_SIZE), b_host.clone()).unwrap();
        let c_nda = a_nda.dot(&b_nda).into_raw_vec();
        assert_eq!(c_nda, c_host, "GPU result does not match CPU impl");
    }

    // Batchsize = 10
    {
        const QUERY_SIZE: usize = 320;
        let a_host = (0..DB_SIZE * WIDTH)
            .map(|_| rng.gen::<u16>() as u16)
            .collect::<Vec<_>>();
        let b_host = (0..QUERY_SIZE * WIDTH)
            .map(|_| rng.gen_range(0..2) as u16)
            .collect::<Vec<_>>();
        let mut c_host = vec![0u16; (DB_SIZE * QUERY_SIZE) as usize];

        // Pin memory if DMA is available
        unsafe {
            cuMemAllocHost_v2(c_host.as_mut_ptr() as *mut _, DB_SIZE * QUERY_SIZE * 2);
        }

        let a_dev = dev.htod_sync_copy(&a_host).unwrap();
        let b_dev = dev.htod_sync_copy(&b_host).unwrap();
        let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

        let cfg = LaunchConfig {
            block_dim: (64, 1, 1), // num_warps = 2 (warp = 32 threads)
            grid_dim: (
                (DB_SIZE.div_ceil(32) * QUERY_SIZE.div_ceil(32)) as u32,
                1,
                1,
            ),
            shared_mem_bytes: 8192,
        };

        group.bench_function("custom kernel triton batchsize 10", |b| {
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
        });

        // Skip sanity check here bc it takes too long
    }

    group.finish();
}

criterion_group!(benches, custom_kernel_triton,);
criterion_main!(benches);
