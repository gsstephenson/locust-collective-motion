# Week 1 — Baseline Data Statistics
## Weinburd et al. (2024) Locust Trajectory Dataset

**Generated:** from `week1_data_pipeline.ipynb`

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Species | Australian plague locust (*Chortoicetes terminifera*) |
| Total clips | 27 |
| Hopper bands | 4 (vid133, vid098, vid096, vid146) |
| Frame rate | 25 fps (dt = 0.04 s) |
| Frames per clip | 1,500 (60 seconds) |
| Total trajectories | 19,687 |
| Total valid observations | 3,362,321 |

## Per-Clip Summary

| Clip | Band | Locusts | Valid Obs | Density (locusts/m²) | Arena (cm) |
|------|------|---------|-----------|---------------------|------------|
| 133_1min_225 | Band 1 (vid133) | 1408 | 197,487 | 236.7 | 99.4 x 55.9 |
| 133_1min_650 | Band 1 (vid133) | 1520 | 220,085 | 263.7 | 99.4 x 55.9 |
| 133_1min_750 | Band 1 (vid133) | 1739 | 229,595 | 275.1 | 99.4 x 55.9 |
| 133_1min_850 | Band 1 (vid133) | 1593 | 207,457 | 248.6 | 99.4 x 55.9 |
| 133_1min_950 | Band 1 (vid133) | 983 | 123,689 | 148.2 | 99.4 x 55.9 |
| 133_1min_1115 | Band 1 (vid133) | 898 | 153,857 | 184.4 | 99.4 x 55.9 |
| 098_1min_430 | Band 3 (vid098) | 715 | 135,146 | 145.2 | 109.7 x 56.5 |
| 098_1min_530 | Band 3 (vid098) | 907 | 155,959 | 167.6 | 109.7 x 56.5 |
| 098_1min_630 | Band 3 (vid098) | 1157 | 175,078 | 188.1 | 109.7 x 56.5 |
| 098_1min_730 | Band 3 (vid098) | 937 | 180,266 | 193.7 | 109.7 x 56.5 |
| 098_1min_5830 | Band 3 (vid098) | 601 | 98,012 | 105.3 | 109.7 x 56.5 |
| 098_1min_5930 | Band 3 (vid098) | 937 | 123,202 | 132.4 | 109.7 x 56.5 |
| 098_1min_10030 | Band 3 (vid098) | 797 | 109,522 | 117.7 | 109.7 x 56.5 |
| 096_1min_500 | Band 2 (vid096) | 569 | 153,962 | 277.2 | 83.1 x 44.6 |
| 096_1min_600 | Band 2 (vid096) | 493 | 126,226 | 227.2 | 83.1 x 44.6 |
| 096_1min_700 | Band 2 (vid096) | 475 | 126,244 | 227.3 | 83.1 x 44.6 |
| 096_1min_800 | Band 2 (vid096) | 515 | 130,117 | 234.3 | 83.1 x 44.6 |
| 096_1min_900 | Band 2 (vid096) | 566 | 133,231 | 239.9 | 83.1 x 44.6 |
| 096_1min_1000 | Band 2 (vid096) | 530 | 125,539 | 226.0 | 83.1 x 44.6 |
| 096_1min_1100 | Band 2 (vid096) | 490 | 124,589 | 224.3 | 83.1 x 44.6 |
| 146_1min_515 | Band 6 (vid146) | 169 | 37,090 | 57.5 | 91.7 x 46.9 |
| 146_1min_615 | Band 6 (vid146) | 115 | 49,689 | 77.1 | 91.7 x 46.9 |
| 146_1min_715 | Band 6 (vid146) | 255 | 52,119 | 80.8 | 91.7 x 46.9 |
| 146_1min_815 | Band 6 (vid146) | 404 | 49,183 | 76.3 | 91.7 x 46.9 |
| 146_1min_915 | Band 6 (vid146) | 181 | 22,718 | 35.2 | 91.7 x 46.9 |
| 146_1min_1015 | Band 6 (vid146) | 286 | 64,818 | 100.5 | 91.7 x 46.9 |
| 146_1min_1115 | Band 6 (vid146) | 447 | 57,441 | 89.1 | 91.7 x 46.9 |

## Metric 1 — Polarization (Order Parameter)

| Statistic | Value |
|-----------|-------|
| Mean | 0.8196 |
| Std | 0.0615 |
| Median | 0.8235 |
| Min | 0.2321 |
| Max | 0.9836 |

## Metric 2 — Speed & Turning Angles

### Speed (cm/s)

| State | Count | Mean | Median | Std |
|-------|-------|------|--------|-----|
| All | 3,326,516 | 6.50 | 2.78 | 8.40 |
| Stationary | 867,450 | 0.18 | 0.04 | 0.70 |
| Walking | 823,073 | 2.74 | 2.30 | 2.22 |
| Hopping | 1,634,212 | 11.76 | 11.53 | 9.21 |

### Turning Angles (rad)

| State | Count | Mean | Std | Circular Std |
|-------|-------|------|-----|-------------|
| All | 3,299,769 | -0.0010 | 0.2599 | — |
| Walking | 816,212 | -0.0015 | 0.3014 | — |
| Hopping | 1,618,401 | -0.0013 | 0.2927 | — |

## Metric 4 — Nearest-Neighbor Distance (cm)

| Statistic | Value |
|-----------|-------|
| Mean | 4.473 |
| Median | 3.893 |
| Std | 2.699 |
| 5th percentile | 1.508 |
| 25th percentile | 2.630 |
| 75th percentile | 5.598 |
| 95th percentile | 9.363 |

## Figures

- `figures/metric1_polarization.png` — Polarization time series, distribution, density-polarization scatter
- `figures/metric2_speed_turning.png` — Speed and turning-angle distributions by motion state
- `figures/metric3_neighbor_density.png` — Body-centered neighbor density maps
- `figures/metric4_nnd.png` — Nearest-neighbor distance distribution and time series
