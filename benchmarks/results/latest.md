# t4perceval vs perception_eval

Generated: `2026-09-08T01:45:11+0900` by `benchmarks/compare.py`.

Both libraries are fed the same synthetic scenes (written once to `.npz`, read by both). Performance uses the `dense` regime; numerical agreement uses the `unambiguous` regime, in which every estimate has exactly one feasible ground truth so greedy and optimal assignment coincide, and the `dense` regime again to show where the two implementations are known to differ.

- CPU: `13th Gen Intel(R) Core(TM) i7-13700F` (one pinned logical CPU)
- Frames per scene: 10; repetitions: 5 (matching: 15) after 2 warm-ups; median shown
- Seed: 0
- `perception_eval`: 1.3.6, Python 3.10.0, NumPy 1.26.4
- `t4perceval`: 0.1.0, Python 3.10.0, NumPy 2.2.6

## Performance

### Construction

NumPy arrays -> one frame of each library's object representation (estimation and ground truth).

| Objects / frame | perception_eval | t4perceval | Speedup |
| --------------: | --------------: | ---------: | ------: |
|              10 |        0.358 ms |   0.062 ms |    5.8x |
|              50 |        1.566 ms |   0.063 ms |   25.0x |
|             100 |        3.015 ms |   0.065 ms |   46.4x |
|             200 |        6.035 ms |   0.069 ms |   87.2x |

### Matching

Center-distance matching of one frame at 1.0 m.

| Objects / frame | perception_eval | t4perceval | Speedup |
| --------------: | --------------: | ---------: | ------: |
|              10 |        7.259 ms |   0.342 ms |   21.2x |
|              50 |       49.233 ms |   0.439 ms |  112.2x |
|             100 |      120.899 ms |   0.696 ms |  173.8x |
|             200 |      339.900 ms |   1.517 ms |  224.1x |

### Detection

mAP and mAPH over the scene at center-distance thresholds [0.5, 1.0, 2.0, 4.0] (matching included).

| Objects / frame | perception_eval | t4perceval | Speedup |
| --------------: | --------------: | ---------: | ------: |
|              10 |      197.235 ms |  39.737 ms |    5.0x |
|              50 |     1246.609 ms |  45.647 ms |   27.3x |
|             100 |     2731.873 ms |  56.916 ms |   48.0x |
|             200 |     6360.713 ms |  94.074 ms |   67.6x |

### Tracking

CLEAR (MOTA / MOTP / ID switches) over the scene at 1.0 m (matching included).

| Objects / frame | perception_eval | t4perceval | Speedup |
| --------------: | --------------: | ---------: | ------: |
|              10 |       66.494 ms |   6.896 ms |    9.6x |
|              50 |      467.072 ms |   8.198 ms |   57.0x |
|             100 |     1165.367 ms |  10.715 ms |  108.8x |
|             200 |     3303.261 ms |  19.573 ms |  168.8x |

### Prediction

ADE / FDE / miss rate over the scene for top-k [1, 3] (matching included).

| Objects / frame | perception_eval | t4perceval | Speedup |
| --------------: | --------------: | ---------: | ------: |
|              10 |       86.787 ms |  13.271 ms |    6.5x |
|              50 |      586.003 ms |  15.586 ms |   37.6x |
|             100 |     1395.468 ms |  19.302 ms |   72.3x |
|             200 |     3718.902 ms |  30.701 ms |  121.1x |

### Retained representation memory

RSS increase after constructing 20,000 estimation and 20,000 ground-truth objects (detection representation):

| Implementation  | RSS increase | Relative usage |
| :-------------- | -----------: | -------------: |
| perception_eval |     83.7 MiB |          16.0x |
| t4perceval      |      5.2 MiB |           1.0x |

## Numerical agreement (unambiguous scene)

50 objects per frame, 10 frames. 133 match, 0 documented divergences, 0 unexplained mismatches.

| Task       | Metric    | Threshold / k | Class      | perception_eval | t4perceval | abs diff | Status |
| :--------- | :-------- | :------------ | :--------- | --------------: | ---------: | -------: | :----- |
| detection  | ap        | 0.5           | ALL\*      |        0.376964 |   0.376964 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | bicycle    |        0.341988 |   0.341988 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | bus        |        0.093747 |   0.093747 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | car        |        0.532413 |   0.532413 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | motorbike  |        0.480746 |   0.480746 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | pedestrian |        0.538551 |   0.538551 | 0.00e+00 | match  |
| detection  | ap        | 0.5           | truck      |        0.274338 |   0.274338 | 0.00e+00 | match  |
| detection  | ap        | 1             | ALL\*      |        0.891555 |   0.891555 | 0.00e+00 | match  |
| detection  | ap        | 1             | bicycle    |        0.844423 |   0.844423 | 0.00e+00 | match  |
| detection  | ap        | 1             | bus        |        0.900500 |   0.900500 | 0.00e+00 | match  |
| detection  | ap        | 1             | car        |        0.892939 |   0.892939 | 0.00e+00 | match  |
| detection  | ap        | 1             | motorbike  |        0.861118 |   0.861118 | 0.00e+00 | match  |
| detection  | ap        | 1             | pedestrian |        0.910072 |   0.910072 | 0.00e+00 | match  |
| detection  | ap        | 1             | truck      |        0.940278 |   0.940278 | 0.00e+00 | match  |
| detection  | ap        | 2             | ALL\*      |        0.891555 |   0.891555 | 0.00e+00 | match  |
| detection  | ap        | 2             | bicycle    |        0.844423 |   0.844423 | 0.00e+00 | match  |
| detection  | ap        | 2             | bus        |        0.900500 |   0.900500 | 0.00e+00 | match  |
| detection  | ap        | 2             | car        |        0.892939 |   0.892939 | 0.00e+00 | match  |
| detection  | ap        | 2             | motorbike  |        0.861118 |   0.861118 | 0.00e+00 | match  |
| detection  | ap        | 2             | pedestrian |        0.910072 |   0.910072 | 0.00e+00 | match  |
| detection  | ap        | 2             | truck      |        0.940278 |   0.940278 | 0.00e+00 | match  |
| detection  | ap        | 4             | ALL\*      |        0.891555 |   0.891555 | 0.00e+00 | match  |
| detection  | ap        | 4             | bicycle    |        0.844423 |   0.844423 | 0.00e+00 | match  |
| detection  | ap        | 4             | bus        |        0.900500 |   0.900500 | 0.00e+00 | match  |
| detection  | ap        | 4             | car        |        0.892939 |   0.892939 | 0.00e+00 | match  |
| detection  | ap        | 4             | motorbike  |        0.861118 |   0.861118 | 0.00e+00 | match  |
| detection  | ap        | 4             | pedestrian |        0.910072 |   0.910072 | 0.00e+00 | match  |
| detection  | ap        | 4             | truck      |        0.940278 |   0.940278 | 0.00e+00 | match  |
| detection  | ap        | mean          | bicycle    |        0.718814 |   0.718814 | 0.00e+00 | match  |
| detection  | ap        | mean          | bus        |        0.698812 |   0.698812 | 0.00e+00 | match  |
| detection  | ap        | mean          | car        |        0.802808 |   0.802808 | 0.00e+00 | match  |
| detection  | ap        | mean          | motorbike  |        0.766025 |   0.766025 | 0.00e+00 | match  |
| detection  | ap        | mean          | pedestrian |        0.817192 |   0.817192 | 0.00e+00 | match  |
| detection  | ap        | mean          | truck      |        0.773793 |   0.773793 | 0.00e+00 | match  |
| detection  | aph       | 0.5           | ALL\*      |        0.358135 |   0.358135 | 9.92e-13 | match  |
| detection  | aph       | 0.5           | bicycle    |        0.324986 |   0.324986 | 1.96e-12 | match  |
| detection  | aph       | 0.5           | bus        |        0.088056 |   0.088056 | 8.81e-15 | match  |
| detection  | aph       | 0.5           | car        |        0.502462 |   0.502462 | 4.29e-13 | match  |
| detection  | aph       | 0.5           | motorbike  |        0.458942 |   0.458942 | 1.34e-12 | match  |
| detection  | aph       | 0.5           | pedestrian |        0.518415 |   0.518415 | 1.97e-13 | match  |
| detection  | aph       | 0.5           | truck      |        0.255952 |   0.255952 | 2.03e-12 | match  |
| detection  | aph       | 1             | ALL\*      |        0.861779 |   0.861779 | 4.53e-14 | match  |
| detection  | aph       | 1             | bicycle    |        0.821592 |   0.821592 | 2.00e-13 | match  |
| detection  | aph       | 1             | bus        |        0.877938 |   0.877938 | 2.26e-13 | match  |
| detection  | aph       | 1             | car        |        0.859366 |   0.859366 | 2.32e-14 | match  |
| detection  | aph       | 1             | motorbike  |        0.828423 |   0.828423 | 2.71e-13 | match  |
| detection  | aph       | 1             | pedestrian |        0.876926 |   0.876926 | 2.89e-15 | match  |
| detection  | aph       | 1             | truck      |        0.906428 |   0.906428 | 0.00e+00 | match  |
| detection  | aph       | 2             | ALL\*      |        0.861779 |   0.861779 | 4.53e-14 | match  |
| detection  | aph       | 2             | bicycle    |        0.821592 |   0.821592 | 2.00e-13 | match  |
| detection  | aph       | 2             | bus        |        0.877938 |   0.877938 | 2.26e-13 | match  |
| detection  | aph       | 2             | car        |        0.859366 |   0.859366 | 2.32e-14 | match  |
| detection  | aph       | 2             | motorbike  |        0.828423 |   0.828423 | 2.71e-13 | match  |
| detection  | aph       | 2             | pedestrian |        0.876926 |   0.876926 | 2.89e-15 | match  |
| detection  | aph       | 2             | truck      |        0.906428 |   0.906428 | 0.00e+00 | match  |
| detection  | aph       | 4             | ALL\*      |        0.861779 |   0.861779 | 4.53e-14 | match  |
| detection  | aph       | 4             | bicycle    |        0.821592 |   0.821592 | 2.00e-13 | match  |
| detection  | aph       | 4             | bus        |        0.877938 |   0.877938 | 2.26e-13 | match  |
| detection  | aph       | 4             | car        |        0.859366 |   0.859366 | 2.32e-14 | match  |
| detection  | aph       | 4             | motorbike  |        0.828423 |   0.828423 | 2.71e-13 | match  |
| detection  | aph       | 4             | pedestrian |        0.876926 |   0.876926 | 2.89e-15 | match  |
| detection  | aph       | 4             | truck      |        0.906428 |   0.906428 | 0.00e+00 | match  |
| detection  | aph       | mean          | bicycle    |        0.697440 |   0.697440 | 6.41e-13 | match  |
| detection  | aph       | mean          | bus        |        0.680467 |   0.680467 | 1.71e-13 | match  |
| detection  | aph       | mean          | car        |        0.770140 |   0.770140 | 1.25e-13 | match  |
| detection  | aph       | mean          | motorbike  |        0.736053 |   0.736053 | 5.38e-13 | match  |
| detection  | aph       | mean          | pedestrian |        0.787298 |   0.787298 | 5.13e-14 | match  |
| detection  | aph       | mean          | truck      |        0.743809 |   0.743809 | 5.08e-13 | match  |
| detection  | map       | -             | ALL        |        0.762907 |   0.762907 | 0.00e+00 | match  |
| detection  | maph      | -             | ALL        |        0.735868 |   0.735868 | 2.82e-13 | match  |
| prediction | ade       | k1            | ALL\*      |        0.709851 |   0.709851 | 0.00e+00 | match  |
| prediction | ade       | k1            | bicycle    |        0.770057 |   0.770057 | 1.11e-16 | match  |
| prediction | ade       | k1            | bus        |        0.707265 |   0.707265 | 1.11e-16 | match  |
| prediction | ade       | k1            | car        |        0.599225 |   0.599225 | 4.44e-16 | match  |
| prediction | ade       | k1            | motorbike  |        0.695325 |   0.695325 | 0.00e+00 | match  |
| prediction | ade       | k1            | pedestrian |        0.660049 |   0.660049 | 1.11e-16 | match  |
| prediction | ade       | k1            | truck      |        0.827186 |   0.827186 | 1.11e-16 | match  |
| prediction | ade       | k3            | ALL\*      |        0.702564 |   0.702564 | 1.11e-16 | match  |
| prediction | ade       | k3            | bicycle    |        0.696591 |   0.696591 | 1.11e-16 | match  |
| prediction | ade       | k3            | bus        |        0.707929 |   0.707929 | 4.44e-16 | match  |
| prediction | ade       | k3            | car        |        0.660055 |   0.660055 | 2.22e-16 | match  |
| prediction | ade       | k3            | motorbike  |        0.728458 |   0.728458 | 1.11e-16 | match  |
| prediction | ade       | k3            | pedestrian |        0.622010 |   0.622010 | 0.00e+00 | match  |
| prediction | ade       | k3            | truck      |        0.800343 |   0.800343 | 1.11e-16 | match  |
| prediction | fde       | k1            | ALL\*      |        1.048503 |   1.048503 | 4.44e-16 | match  |
| prediction | fde       | k1            | bicycle    |        1.167413 |   1.167413 | 0.00e+00 | match  |
| prediction | fde       | k1            | bus        |        0.997277 |   0.997277 | 1.11e-16 | match  |
| prediction | fde       | k1            | car        |        0.887148 |   0.887148 | 0.00e+00 | match  |
| prediction | fde       | k1            | motorbike  |        1.029182 |   1.029182 | 4.44e-16 | match  |
| prediction | fde       | k1            | pedestrian |        1.040865 |   1.040865 | 4.44e-16 | match  |
| prediction | fde       | k1            | truck      |        1.169134 |   1.169134 | 4.44e-16 | match  |
| prediction | fde       | k3            | ALL\*      |        1.034948 |   1.034948 | 0.00e+00 | match  |
| prediction | fde       | k3            | bicycle    |        1.033994 |   1.033994 | 2.22e-16 | match  |
| prediction | fde       | k3            | bus        |        0.985913 |   0.985913 | 6.66e-16 | match  |
| prediction | fde       | k3            | car        |        1.003765 |   1.003765 | 2.22e-16 | match  |
| prediction | fde       | k3            | motorbike  |        1.079598 |   1.079598 | 4.44e-16 | match  |
| prediction | fde       | k3            | pedestrian |        0.969288 |   0.969288 | 0.00e+00 | match  |
| prediction | fde       | k3            | truck      |        1.137128 |   1.137128 | 6.66e-16 | match  |
| prediction | miss_rate | k1            | ALL\*      |        0.014644 |   0.014644 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | bicycle    |        0.018779 |   0.018779 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | bus        |        0.002222 |   0.002222 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | car        |        0.010040 |   0.010040 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | motorbike  |        0.015982 |   0.015982 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | pedestrian |        0.004444 |   0.004444 | 0.00e+00 | match  |
| prediction | miss_rate | k1            | truck      |        0.036398 |   0.036398 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | ALL\*      |        0.015370 |   0.015370 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | bicycle    |        0.014085 |   0.014085 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | bus        |        0.006667 |   0.006667 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | car        |        0.012718 |   0.012718 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | motorbike  |        0.022070 |   0.022070 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | pedestrian |        0.006667 |   0.006667 | 0.00e+00 | match  |
| prediction | miss_rate | k3            | truck      |        0.030013 |   0.030013 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | ALL\*      |        5.000000 |   5.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | bicycle    |        0.000000 |   0.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | bus        |        3.000000 |   3.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | car        |        1.000000 |   1.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | motorbike  |        0.000000 |   0.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | pedestrian |        1.000000 |   1.000000 | 0.00e+00 | match  |
| tracking   | id_switch | 1             | truck      |        0.000000 |   0.000000 | 0.00e+00 | match  |
| tracking   | mota      | 1             | ALL\*      |        0.758000 |   0.758000 | 0.00e+00 | match  |
| tracking   | mota      | 1             | bicycle    |        0.712500 |   0.712500 | 0.00e+00 | match  |
| tracking   | mota      | 1             | bus        |        0.725000 |   0.725000 | 0.00e+00 | match  |
| tracking   | mota      | 1             | car        |        0.755556 |   0.755556 | 0.00e+00 | match  |
| tracking   | mota      | 1             | motorbike  |        0.737500 |   0.737500 | 0.00e+00 | match  |
| tracking   | mota      | 1             | pedestrian |        0.787500 |   0.787500 | 0.00e+00 | match  |
| tracking   | mota      | 1             | truck      |        0.822222 |   0.822222 | 0.00e+00 | match  |
| tracking   | motp      | 1             | ALL\*      |        0.382906 |   0.382906 | 5.55e-17 | match  |
| tracking   | motp      | 1             | bicycle    |        0.378527 |   0.378527 | 5.55e-17 | match  |
| tracking   | motp      | 1             | bus        |        0.543553 |   0.543553 | 1.11e-16 | match  |
| tracking   | motp      | 1             | car        |        0.335839 |   0.335839 | 0.00e+00 | match  |
| tracking   | motp      | 1             | motorbike  |        0.328765 |   0.328765 | 5.55e-17 | match  |
| tracking   | motp      | 1             | pedestrian |        0.267883 |   0.267883 | 5.55e-17 | match  |
| tracking   | motp      | 1             | truck      |        0.437478 |   0.437478 | 5.55e-17 | match  |

## Divergence report (dense scene)

50 objects per frame, 10 frames. 46 match, 87 documented divergences, 0 unexplained mismatches.

| Task       | Metric    | Threshold / k | Class      | perception_eval | t4perceval | abs diff | Status     | Reason                                                                        |
| :--------- | :-------- | :------------ | :--------- | --------------: | ---------: | -------: | :--------- | :---------------------------------------------------------------------------- |
| detection  | ap        | 1             | ALL\*      |        0.632612 |   0.627291 | 5.32e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 1             | motorbike  |        0.593556 |   0.572508 | 2.10e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 1             | truck      |        0.697245 |   0.686364 | 1.09e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 2             | ALL\*      |        0.685764 |   0.682611 | 3.15e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 2             | car        |        0.608602 |   0.602422 | 6.18e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 2             | motorbike  |        0.698879 |   0.696291 | 2.59e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 2             | truck      |        0.733966 |   0.723817 | 1.01e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 4             | ALL\*      |        0.720981 |   0.708777 | 1.22e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 4             | bicycle    |        0.736087 |   0.734044 | 2.04e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 4             | bus        |        0.717173 |   0.702125 | 1.50e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 4             | motorbike  |        0.742436 |   0.730050 | 1.24e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | 4             | truck      |        0.767565 |   0.723817 | 4.37e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | mean          | bicycle    |        0.533987 |   0.533476 | 5.11e-04 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | mean          | bus        |        0.606004 |   0.602242 | 3.76e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | mean          | car        |        0.510014 |   0.508469 | 1.54e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | mean          | motorbike  |        0.537847 |   0.528841 | 9.01e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | ap        | mean          | truck      |        0.586797 |   0.570603 | 1.62e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 0.5           | ALL\*      |        0.168481 |   0.166378 | 2.10e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 0.5           | bicycle    |        0.105914 |   0.104418 | 1.50e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 0.5           | motorbike  |        0.102325 |   0.097980 | 4.34e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 0.5           | pedestrian |        0.215049 |   0.208416 | 6.63e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 0.5           | truck      |        0.116513 |   0.116370 | 1.43e-04 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | ALL\*      |        0.568658 |   0.559980 | 8.68e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | bicycle    |        0.525468 |   0.516201 | 9.27e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | bus        |        0.638121 |   0.638127 | 6.56e-06 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | car        |        0.490671 |   0.489362 | 1.31e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | motorbike  |        0.538285 |   0.518312 | 2.00e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | pedestrian |        0.606571 |   0.596602 | 9.97e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 1             | truck      |        0.612835 |   0.601274 | 1.16e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | ALL\*      |        0.616385 |   0.609622 | 6.76e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | bicycle    |        0.608404 |   0.607554 | 8.50e-04 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | bus        |        0.638121 |   0.638127 | 6.56e-06 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | car        |        0.544033 |   0.537506 | 6.53e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | motorbike  |        0.626740 |   0.623532 | 3.21e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | pedestrian |        0.623393 |   0.613235 | 1.02e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 2             | truck      |        0.657620 |   0.637777 | 1.98e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | ALL\*      |        0.641121 |   0.627988 | 1.31e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | bicycle    |        0.650682 |   0.637402 | 1.33e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | bus        |        0.651886 |   0.638127 | 1.38e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | car        |        0.584613 |   0.584835 | 2.22e-04 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | motorbike  |        0.654796 |   0.643426 | 1.14e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | pedestrian |        0.627042 |   0.626360 | 6.82e-04 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | 4             | truck      |        0.677710 |   0.637777 | 3.99e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | bicycle    |        0.472617 |   0.466394 | 6.22e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | bus        |        0.550036 |   0.546600 | 3.44e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | car        |        0.454596 |   0.452693 | 1.90e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | motorbike  |        0.480536 |   0.470813 | 9.72e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | pedestrian |        0.518014 |   0.511153 | 6.86e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | aph       | mean          | truck      |        0.516169 |   0.498300 | 1.79e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | map       | -             | ALL        |        0.558686 |   0.553517 | 5.17e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| detection  | maph      | -             | ALL        |        0.498662 |   0.490992 | 7.67e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k1            | ALL\*      |        0.407643 |   0.432763 | 2.51e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k1            | bus        |        0.355287 |   0.452553 | 9.73e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k1            | car        |        0.352784 |   0.406237 | 5.35e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k3            | ALL\*      |        0.406816 |   0.431113 | 2.43e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k3            | bus        |        0.357055 |   0.448774 | 9.17e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | ade       | k3            | car        |        0.351712 |   0.405776 | 5.41e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k1            | ALL\*      |        0.674339 |   0.716480 | 4.21e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k1            | bus        |        0.593822 |   0.752573 | 1.59e-01 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k1            | car        |        0.589044 |   0.683139 | 9.41e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k3            | ALL\*      |        0.688972 |   0.727453 | 3.85e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k3            | bus        |        0.591935 |   0.730237 | 1.38e-01 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | fde       | k3            | car        |        0.587193 |   0.679777 | 9.26e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k1            | ALL\*      |        0.008647 |   0.015438 | 6.79e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k1            | bus        |        0.000000 |   0.024876 | 2.49e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k1            | car        |        0.002646 |   0.018519 | 1.59e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k3            | ALL\*      |        0.008505 |   0.015296 | 6.79e-03 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k3            | bus        |        0.000000 |   0.024876 | 2.49e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| prediction | miss_rate | k3            | car        |        0.000882 |   0.016755 | 1.59e-02 | documented | hungarian-vs-greedy (docs/TODOs/metrics.md #3)                                |
| tracking   | id_switch | 1             | ALL\*      |       11.000000 |  18.000000 | 7.00e+00 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | id_switch | 1             | bus        |        1.000000 |   5.000000 | 4.00e+00 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | id_switch | 1             | car        |        0.000000 |   4.000000 | 4.00e+00 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | id_switch | 1             | pedestrian |        5.000000 |   4.000000 | 1.00e+00 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | ALL\*      |        0.572000 |   0.468000 | 1.04e-01 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | bicycle    |        0.587500 |   0.487500 | 1.00e-01 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | bus        |        0.650000 |   0.487500 | 1.63e-01 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | car        |        0.455556 |   0.333333 | 1.22e-01 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | motorbike  |        0.487500 |   0.412500 | 7.50e-02 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | pedestrian |        0.562500 |   0.512500 | 5.00e-02 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | mota      | 1             | truck      |        0.688889 |   0.577778 | 1.11e-01 | documented | idsw-across-missed-frame, idsw-estimation-side (docs/TODOs/metrics.md #4, #8) |
| tracking   | motp      | 1             | ALL\*      |        0.450919 |   0.459187 | 8.27e-03 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | bicycle    |        0.446841 |   0.477781 | 3.09e-02 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | bus        |        0.479838 |   0.456529 | 2.33e-02 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | car        |        0.397113 |   0.404322 | 7.21e-03 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | motorbike  |        0.499978 |   0.496681 | 3.30e-03 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | pedestrian |        0.398366 |   0.444679 | 4.63e-02 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |
| tracking   | motp      | 1             | truck      |        0.479472 |   0.475195 | 4.28e-03 | documented | motp-previous-score (docs/TODOs/metrics.md #6)                                |

## Notes

- Timed t4perceval pipeline phases rebuild a `Store` from prebuilt input chunks on every call (a few microseconds), because a second run would append metric rows at the same reporting time.
- Rows marked `*` are not reported directly by one library and were derived with the other's aggregation formula (perception_eval has no per-threshold all-class AP; t4perceval reports CLEAR and displacement per class only).
- perception_eval's `inf` sentinels (CLEAR with no ground truth or no true positive) are read as `nan`.
- Documented divergences refer to the numbered items in `docs/TODOs/metrics.md`. Only the dense scene may exercise them; a difference in the unambiguous scene is a mismatch.
- perception_eval matches greedily in confidence order and pairs objects across labels in a second pass; t4perceval solves a linear-sum assignment per frame. The unambiguous scene is constructed so both pick the same pairs.
