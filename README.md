# Benchmark Results

## H100

```
legacy cublas u14/leagcy cublas u14 mul with int8 200000 x 310
                        time:   [14.378 ms 14.404 ms 14.434 ms]
                        thrpt:  [138.56 Melem/s 138.85 Melem/s 139.11 Melem/s]
                 change:
                        time:   [+5.6520% +6.5157% +7.2178%] (p = 0.00 < 0.05)
                        thrpt:  [-6.7319% -6.1171% -5.3497%]
                        Performance has regressed.
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) high mild
  3 (3.00%) high severe

bench_u16/u16 x u16 → u16 (200000 x 31)
                        time:   [9.0058 ms 9.0138 ms 9.0259 ms]
                        thrpt:  [22.159 Melem/s 22.188 Melem/s 22.208 Melem/s]
                 change:
                        time:   [+0.4240% +0.7646% +1.0646%] (p = 0.00 < 0.05)
                        thrpt:  [-1.0534% -0.7588% -0.4222%]
                        Change within noise threshold.
Found 3 outliers among 100 measurements (3.00%)
  1 (1.00%) high mild
  2 (2.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 155)
                        time:   [11.606 ms 11.663 ms 11.736 ms]
                        thrpt:  [85.208 Melem/s 85.739 Melem/s 86.165 Melem/s]
                 change:
                        time:   [+4.0447% +4.5851% +5.2045%] (p = 0.00 < 0.05)
                        thrpt:  [-4.9470% -4.3841% -3.8874%]
                        Performance has regressed.
Found 11 outliers among 100 measurements (11.00%)
  1 (1.00%) low mild
  1 (1.00%) high mild
  9 (9.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 310)
                        time:   [14.114 ms 14.141 ms 14.173 ms]
                        thrpt:  [141.11 Melem/s 141.43 Melem/s 141.70 Melem/s]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 620)
                        time:   [30.716 ms 30.760 ms 30.810 ms]
                        thrpt:  [129.83 Melem/s 130.04 Melem/s 130.23 Melem/s]
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 930)
                        time:   [36.704 ms 36.990 ms 37.350 ms]
                        thrpt:  [160.64 Melem/s 162.20 Melem/s 163.47 Melem/s]
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 1550)
                        time:   [69.576 ms 70.461 ms 71.541 ms]
                        thrpt:  [139.78 Melem/s 141.92 Melem/s 143.73 Melem/s]
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) high mild
  6 (6.00%) high severe
bench_u16/u16 x u16 → u16 (200000 x 3100)
                        time:   [130.32 ms 131.31 ms 132.45 ms]
                        thrpt:  [151.00 Melem/s 152.31 Melem/s 153.47 Melem/s]
Found 6 outliers among 100 measurements (6.00%)
  3 (3.00%) high mild
  3 (3.00%) high severe

bench_p16/p16 x p16 → p16 (200000 x 31)
                        time:   [11.920 ms 11.922 ms 11.924 ms]
                        thrpt:  [16.774 Melem/s 16.776 Melem/s 16.779 Melem/s]
bench_p16/p16 x p16 → p16 (200000 x 155)
                        time:   [14.086 ms 14.097 ms 14.110 ms]
                        thrpt:  [70.874 Melem/s 70.936 Melem/s 70.993 Melem/s]
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe
bench_p16/p16 x p16 → p16 (200000 x 310)
                        time:   [15.982 ms 16.003 ms 16.028 ms]
                        thrpt:  [124.78 Melem/s 124.98 Melem/s 125.14 Melem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high severe
bench_p16/p16 x p16 → p16 (200000 x 620)
                        time:   [35.594 ms 35.642 ms 35.696 ms]
                        thrpt:  [112.06 Melem/s 112.23 Melem/s 112.38 Melem/s]
Found 10 outliers among 100 measurements (10.00%)
  9 (9.00%) high mild
  1 (1.00%) high severe
bench_p16/p16 x p16 → p16 (200000 x 930)
                        time:   [40.465 ms 40.618 ms 40.791 ms]
                        thrpt:  [147.09 Melem/s 147.72 Melem/s 148.28 Melem/s]
Found 16 outliers among 100 measurements (16.00%)
  5 (5.00%) high mild
  11 (11.00%) high severe
bench_p16/p16 x p16 → p16 (200000 x 1550)
                        time:   [77.453 ms 78.743 ms 80.239 ms]
                        thrpt:  [124.63 Melem/s 127.00 Melem/s 129.11 Melem/s]
Found 11 outliers among 100 measurements (11.00%)
  6 (6.00%) high mild
  5 (5.00%) high severe
bench_p16/p16 x p16 → p16 (200000 x 3100)
                        time:   [145.79 ms 147.64 ms 150.01 ms]
                        thrpt:  [133.33 Melem/s 135.46 Melem/s 137.18 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) high mild
  6 (6.00%) high severe

bench_u32/u32 x u32 → u32 (200000 x 31)
                        time:   [12.353 ms 12.389 ms 12.439 ms]
                        thrpt:  [16.078 Melem/s 16.143 Melem/s 16.190 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe
bench_u32/u32 x u32 → u32 (200000 x 155)
                        time:   [17.379 ms 17.398 ms 17.422 ms]
                        thrpt:  [57.399 Melem/s 57.479 Melem/s 57.541 Melem/s]
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) low mild
  1 (1.00%) high severe
bench_u32/u32 x u32 → u32 (200000 x 310)
                        time:   [23.586 ms 23.616 ms 23.646 ms]
                        thrpt:  [84.580 Melem/s 84.688 Melem/s 84.796 Melem/s]
bench_u32/u32 x u32 → u32 (200000 x 620)
                        time:   [55.741 ms 57.330 ms 59.124 ms]
                        thrpt:  [67.655 Melem/s 69.771 Melem/s 71.761 Melem/s]
Found 6 outliers among 100 measurements (6.00%)
  2 (2.00%) high mild
  4 (4.00%) high severe
bench_u32/u32 x u32 → u32 (200000 x 930)
                        time:   [65.854 ms 66.806 ms 68.058 ms]
                        thrpt:  [88.160 Melem/s 89.813 Melem/s 91.111 Melem/s]
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) high mild
  3 (3.00%) high severe
bench_u32/u32 x u32 → u32 (200000 x 1550)
                        time:   [117.38 ms 118.97 ms 120.99 ms]
                        thrpt:  [82.655 Melem/s 84.057 Melem/s 85.192 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe
bench_u32/u32 x u32 → u32 (200000 x 3100)
                        time:   [232.52 ms 234.09 ms 235.85 ms]
                        thrpt:  [84.799 Melem/s 85.437 Melem/s 86.015 Melem/s]
Found 6 outliers among 100 measurements (6.00%)
  2 (2.00%) high mild
  4 (4.00%) high severe

bench_p14/p14 x p14 → p14 (200000 x 31)
                        time:   [9.2862 ms 9.3003 ms 9.3196 ms]
                        thrpt:  [21.460 Melem/s 21.505 Melem/s 21.537 Melem/s]
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) low mild
  3 (3.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 155)
                        time:   [11.628 ms 11.711 ms 11.856 ms]
                        thrpt:  [84.343 Melem/s 85.393 Melem/s 86.000 Melem/s]
Found 9 outliers among 100 measurements (9.00%)
  3 (3.00%) low mild
  1 (1.00%) high mild
  5 (5.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 310)
                        time:   [15.219 ms 15.730 ms 16.358 ms]
                        thrpt:  [122.27 Melem/s 127.15 Melem/s 131.41 Melem/s]
Found 13 outliers among 100 measurements (13.00%)
  13 (13.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 620)
                        time:   [31.986 ms 32.418 ms 32.974 ms]
                        thrpt:  [121.31 Melem/s 123.39 Melem/s 125.05 Melem/s]
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) high mild
  6 (6.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 930)
                        time:   [38.559 ms 39.455 ms 40.540 ms]
                        thrpt:  [148.00 Melem/s 152.07 Melem/s 155.61 Melem/s]
Found 15 outliers among 100 measurements (15.00%)
  2 (2.00%) high mild
  13 (13.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 1550)
                        time:   [70.519 ms 71.149 ms 71.912 ms]
                        thrpt:  [139.06 Melem/s 140.55 Melem/s 141.81 Melem/s]
Found 11 outliers among 100 measurements (11.00%)
  4 (4.00%) high mild
  7 (7.00%) high severe
bench_p14/p14 x p14 → p14 (200000 x 3100)
                        time:   [140.96 ms 143.21 ms 145.79 ms]
                        thrpt:  [137.19 Melem/s 139.66 Melem/s 141.88 Melem/s]
Found 15 outliers among 100 measurements (15.00%)
  3 (3.00%) high mild
  12 (12.00%) high severe

bench_u14/u14 x u14 → u14 (200000 x 31)
                        time:   [8.9675 ms 8.9802 ms 8.9964 ms]
                        thrpt:  [22.231 Melem/s 22.271 Melem/s 22.303 Melem/s]
Found 8 outliers among 100 measurements (8.00%)
  4 (4.00%) high mild
  4 (4.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 155)
                        time:   [11.651 ms 11.696 ms 11.756 ms]
                        thrpt:  [85.063 Melem/s 85.496 Melem/s 85.832 Melem/s]
Found 14 outliers among 100 measurements (14.00%)
  2 (2.00%) low mild
  6 (6.00%) high mild
  6 (6.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 310)
                        time:   [14.523 ms 14.582 ms 14.652 ms]
                        thrpt:  [136.50 Melem/s 137.15 Melem/s 137.71 Melem/s]
Found 12 outliers among 100 measurements (12.00%)
  6 (6.00%) low mild
  1 (1.00%) high mild
  5 (5.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 620)
                        time:   [31.710 ms 31.764 ms 31.828 ms]
                        thrpt:  [125.68 Melem/s 125.93 Melem/s 126.14 Melem/s]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 930)
                        time:   [36.817 ms 36.875 ms 36.946 ms]
                        thrpt:  [162.40 Melem/s 162.71 Melem/s 162.97 Melem/s]
Found 6 outliers among 100 measurements (6.00%)
  2 (2.00%) high mild
  4 (4.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 1550)
                        time:   [69.945 ms 71.056 ms 72.408 ms]
                        thrpt:  [138.11 Melem/s 140.73 Melem/s 142.97 Melem/s]
Found 12 outliers among 100 measurements (12.00%)
  4 (4.00%) high mild
  8 (8.00%) high severe
bench_u14/u14 x u14 → u14 (200000 x 3100)
                        time:   [135.79 ms 137.44 ms 139.19 ms]
                        thrpt:  [143.69 Melem/s 145.52 Melem/s 147.28 Melem/s]
Found 9 outliers among 100 measurements (9.00%)
  9 (9.00%) high mild
```
