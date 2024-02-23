# Benchmark Results

## RTX 4090

```

     Running benches/cublas.rs (target/release/deps/cublas-2572128c9942af20)
cublas/cublas int8 100000 x 31
                        time:   [3.2156 ms 3.2193 ms 3.2243 ms]
                        change: [+0.0583% +0.1756% +0.3340%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 16 outliers among 100 measurements (16.00%)
  1 (1.00%) high mild   
  15 (15.00%) high severe
cublas/cublas fp32 100000 x 31
                        time:   [6.4945 ms 6.4948 ms 6.4951 ms]
                        change: [-0.0110% -0.0053% +0.0006%] (p = 0.09 > 0.05)
                        No change in performance detected.
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe
Benchmarking cublas/cublas fp64 100000 x 31: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 6.9s, or reduce sample count to 70.
cublas/cublas fp64 100000 x 31
                        time:   [68.826 ms 68.827 ms 68.827 ms]
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild

     Running benches/custom_kernel.rs (target/release/deps/custom_kernel-b04de854a367aab0)
custom_kernel/custom kernel naïve 100000 x 32
                        time:   [14.815 ms 14.820 ms 14.825 ms]
                        change: [-0.3506% -0.3178% -0.2801%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 18 outliers among 100 measurements (18.00%)
  18 (18.00%) high severe

     Running benches/custom_kernel_2d.rs (target/release/deps/custom_kernel_2d-d3d72b84852609a3)
custom_kernel/custom kernel 2d 100000 x 32
                        time:   [11.639 ms 11.644 ms 11.648 ms]
                        change: [-0.0377% +0.0146% +0.0646%] (p = 0.57 > 0.05)
                        No change in performance detected.

     Running benches/custom_kernel_triton.rs (target/release/deps/custom_kernel_triton-6c97decae1f8abcc)
custom_kernel/custom kernel triton 100000 x 32 batchsize 1
                        time:   [6.1211 ms 6.1237 ms 6.1269 ms]
Found 11 outliers among 100 measurements (11.00%)
  2 (2.00%) high mild
  9 (9.00%) high severe
custom_kernel/custom kernel triton batchsize 10
                        time:   [33.448 ms 33.544 ms 33.640 ms]
                        change: [-0.4540% -0.0614% +0.3095%] (p = 0.76 > 0.05)
                        No change in performance detected.

```

## A100 (try #1)

```
    Running benches/cublas.rs (target/release/deps/cublas-2572128c9942af20)
cublas/cublas int8 100000 x 31
                        time:   [6.6696 ms 6.7082 ms 6.7477 ms]
                        change: [+19.236% +19.936% +20.605%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) low mild
cublas/cublas fp32 100000 x 31
                        time:   [6.6640 ms 6.7126 ms 6.7582 ms]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) low mild
cublas/cublas fp64 100000 x 31
                        time:   [9.9811 ms 10.054 ms 10.122 ms]
Found 4 outliers among 100 measurements (4.00%)
  3 (3.00%) low mild
  1 (1.00%) high mild

     Running benches/custom_kernel.rs (target/release/deps/custom_kernel-b04de854a367aab0)
custom_kernel/custom kernel naïve 100000 x 32
                        time:   [19.525 ms 19.579 ms 19.619 ms]
Found 4 outliers among 100 measurements (4.00%)
  2 (2.00%) low severe
  1 (1.00%) high mild
  1 (1.00%) high severe

     Running benches/custom_kernel_2d.rs (target/release/deps/custom_kernel_2d-d3d72b84852609a3)
custom_kernel/custom kernel 2d 100000 x 32
                        time:   [14.816 ms 14.889 ms 14.952 ms]
Found 7 outliers among 100 measurements (7.00%)
  4 (4.00%) low severe
  1 (1.00%) low mild
  2 (2.00%) high mild
```