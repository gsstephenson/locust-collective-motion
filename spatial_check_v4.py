"""Body-centered neighbor density at the v4 CMA-ES optimum.

Mirrors spatial_check.py but for hybrid v4 (no temporal integration).
Outputs week5_cma/v4_metric3_density.npy for the near-zone analysis
in fig4."""

import json
from pathlib import Path

import numpy as np

from hybrid_v4 import V3_DEFAULTS, run_hybrid_v4

BASE   = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
OUTDIR = BASE / 'week5_cma'
BEST   = json.load(open(OUTDIR / 'full_best.json'))['best']['params']
print('v4 optimum:', BEST)


def neighbor_density_map(ph, hh, Lx, Ly, radius=10.0, nbins=50,
                         subsample=5, burn_in=500):
    edges = np.linspace(-radius, radius, nbins + 1)
    hist  = np.zeros((nbins, nbins))
    for t in range(burn_in, len(ph), subsample):
        pos, theta = ph[t], hh[t]
        d = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        d[:, :, 0] -= Lx * np.round(d[:, :, 0] / Lx)
        d[:, :, 1] -= Ly * np.round(d[:, :, 1] / Ly)
        dd = np.hypot(d[:, :, 0], d[:, :, 1])
        for i in range(len(pos)):
            m = (dd[i] > 0.1) & (dd[i] < radius)
            if not m.any():
                continue
            c, s = np.cos(theta[i]), np.sin(theta[i])
            rx =  c * d[i, m, 0] + s * d[i, m, 1]
            ry = -s * d[i, m, 0] + c * d[i, m, 1]
            h, _, _ = np.histogram2d(rx, ry, bins=[edges, edges])
            hist += h
    return hist, edges


N_SEEDS = 5
agg = None
print(f'Running {N_SEEDS} seeds at v4 optimum...')
for seed in range(N_SEEDS):
    full = {**V3_DEFAULTS, **BEST}
    ph, hh = run_hybrid_v4(**full, seed=seed)
    h, _   = neighbor_density_map(ph, hh, full['Lx'], full['Ly'])
    print(f'  seed {seed}: {int(h.sum())} neighbor counts')
    agg = h if agg is None else agg + h

np.save(OUTDIR / 'v4_metric3_density.npy', agg)
print(f'Saved: {OUTDIR / "v4_metric3_density.npy"}')
