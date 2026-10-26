# Benchmarks (n = 50,000)

| Implementation | Best | Median | Speed-up |
|---|---:|---:|---:|
| std: loop | 5.41 ms | 5.47 ms | 1.0x |
| std: numcompute | 426 µs | 463 µs | 12.7x |
| histogram (20 bins): loop | 7.88 ms | 8.19 ms | 1.0x |
| histogram (20 bins): numcompute | 1.64 ms | 1.65 ms | 4.8x |
| StandardScaler: loop | 6.06 ms | 6.2 ms | 1.0x |
| StandardScaler: numcompute | 448 µs | 454 µs | 13.5x |
| distances (200 x 8): loop | 5.93 ms | 5.98 ms | 1.0x |
| distances (200 x 8): numcompute | 181 µs | 184 µs | 32.7x |
| top-10: loop (heapq) | 643 µs | 681 µs | 1.0x |
| top-10: numcompute | 468 µs | 478 µs | 1.4x |
| median: loop (sorted) | 5.37 ms | 5.39 ms | 1.0x |
| median: quickselect | 1.01 ms | 1.02 ms | 5.3x |
| rank (average, ties): loop | 20.1 ms | 20.2 ms | 1.0x |
| rank (average, ties): numcompute | 4.06 ms | 4.25 ms | 4.9x |

Python 3.14.7, NumPy 2.5.4, Windows-11-10.0.26300-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
