# Benchmarks (n = 50,000)

| Implementation | Best | Median | Speed-up |
|---|---:|---:|---:|
| std: loop | 5.6 ms | 5.67 ms | 1.0x |
| std: numcompute | 404 µs | 500 µs | 13.8x |
| histogram (20 bins): loop | 11.9 ms | 15 ms | 1.0x |
| histogram (20 bins): numcompute | 2.58 ms | 2.62 ms | 4.6x |
| StandardScaler: loop | 7.86 ms | 8.32 ms | 1.0x |
| StandardScaler: numcompute | 600 µs | 867 µs | 13.1x |
| distances (200 x 8): loop | 9.55 ms | 11.7 ms | 1.0x |
| distances (200 x 8): numcompute | 293 µs | 302 µs | 32.6x |
| top-10: loop (heapq) | 867 µs | 1.19 ms | 1.0x |
| top-10: numcompute | 669 µs | 787 µs | 1.3x |
| median: loop (sorted) | 9.48 ms | 11.7 ms | 1.0x |
| median: quickselect | 1.41 ms | 1.74 ms | 6.7x |
| rank (average, ties): loop | 33.2 ms | 37.8 ms | 1.0x |
| rank (average, ties): numcompute | 4.78 ms | 6.61 ms | 7.0x |

Python 3.14.7, NumPy 2.5.4, Windows-11-10.0.26300-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
