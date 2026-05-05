"""
Spatial-physics check at the v5 CMA-ES optimum.

CMA-ES calibrated against three SCALAR metrics — polarization, turning-angle
std, NND. The original physical motivation (slide 8) was the FORWARD VOID:
a body-centered neighbor density depleted in front of the focal agent.
This script renders that map at the v5 optimum to confirm the spatial
anisotropy survives the calibration.

Outputs:
    week5_cma/v5_metric3_neighbor_density.png
    week5_cma/v5_metric3_density.npy
    week5_cma/v5_anisotropy_index.json
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from hybrid_v5 import V5_DEFAULTS, run_hybrid_v5

BASE   = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
OUTDIR = BASE / 'week5_cma'
BEST_JSON  = OUTDIR / 'v5_full_best.json'
FIELD_PNG  = BASE / 'week1' / 'figures' / 'metric3_neighbor_density.png'

with open(BEST_JSON) as f:
    best = json.load(f)
p_star = best['best']['params']
print(f'Loaded v5 optimum (loss={best["best"]["loss"]:.4f}):')
for k, v in p_star.items():
    print(f'  {k:<22} = {v:.4f}')


def neighbor_density_map(ph, hh, Lx, Ly, radius=10.0, nbins=50, subsample=5,
                         burn_in=500):
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
            rel = d[i, m]
            r   = np.pi/2 - theta[i]
            rx  = rel[:, 0] * np.cos(r) - rel[:, 1] * np.sin(r)
            ry  = rel[:, 0] * np.sin(r) + rel[:, 1] * np.cos(r)
            h, _, _ = np.histogram2d(rx, ry, bins=edges)
            hist += h
    return hist, edges


def anisotropy_index(density):
    """A = (rho_rear - rho_front) / (rho_rear + rho_front).
    Body-centered map: 'ahead' = +y, so the upper half of the array
    (high y indices) corresponds to in front of the focal agent."""
    H = density.shape[1]   # y is axis 1 in (rx, ry) histogram2d output
    cy = H // 2
    rho_rear  = density[:, :cy].mean()   # y < 0
    rho_front = density[:, cy:].mean()   # y > 0
    denom = rho_rear + rho_front
    return float((rho_rear - rho_front) / denom) if denom > 0 else 0.0


# Run multiple seeds and accumulate the density map
N_SEEDS = 5
agg = None
edges_ref = None
print(f'\nRunning {N_SEEDS} seeds at v5 optimum to build density map...')
for seed in range(N_SEEDS):
    full = {**V5_DEFAULTS, **p_star}
    ph, hh = run_hybrid_v5(**full, seed=seed)
    h, edges = neighbor_density_map(ph, hh, full['Lx'], full['Ly'])
    print(f'  seed {seed}: {int(h.sum())} neighbor counts')
    agg = h if agg is None else agg + h
    edges_ref = edges
density = agg / agg.sum()

# Anisotropy index
A = anisotropy_index(agg)
print(f'\nAnisotropy index A = {A:+.4f}')
print('  A > 0 → forward void (more density behind than ahead)')
print('  A = 0 → isotropic')
print('  A < 0 → forward bunching')

# Save raw density
np.save(OUTDIR / 'v5_metric3_density.npy', agg)

# ── Figure: side-by-side with field ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Field reference panel
if FIELD_PNG.exists():
    axes[0].imshow(np.asarray(Image.open(FIELD_PNG)))
    axes[0].set_title('Field (Weinburd 2024)\nforward void present',
                      fontsize=12)
    axes[0].axis('off')
else:
    axes[0].text(0.5, 0.5, '(field reference png not found)',
                 ha='center', va='center', transform=axes[0].transAxes)
    axes[0].axis('off')

# v5 calibrated panel
ax = axes[1]
im = ax.imshow(density.T, origin='lower',
               extent=[edges_ref[0], edges_ref[-1], edges_ref[0], edges_ref[-1]],
               cmap='hot', aspect='equal')
ax.plot(0, 0, 'w^', markersize=14)
ax.axhline(0, color='white', ls='--', alpha=0.3)
ax.axvline(0, color='white', ls='--', alpha=0.3)
ax.set_xlabel('Left ← → Right (cm)')
ax.set_ylabel('Behind ← → Ahead (cm)')
verdict = ('forward VOID present' if A >  0.05 else
           'forward BUNCHING'      if A < -0.05 else
           'isotropic')
ax.set_title(f'v5 calibrated optimum\n'
             f'A = {A:+.3f}  ({verdict})\n'
             f'η={p_star["eta_base"]:.2f}, λ={p_star["lambda_pull"]:.2f}, '
             f'α={p_star["alpha_aniso"]:.2f}',
             fontsize=11)
plt.colorbar(im, ax=ax, label='Relative density')

fig.suptitle('Metric 3 spatial check — does the calibrated model still produce the void?',
             fontsize=13, y=1.02)
plt.tight_layout()

out_png = OUTDIR / 'v5_metric3_neighbor_density.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_png}')

with open(OUTDIR / 'v5_anisotropy_index.json', 'w') as f:
    json.dump({
        'anisotropy_index_A': A,
        'verdict': verdict,
        'n_seeds': N_SEEDS,
        'params': p_star,
    }, f, indent=2)
print(f'Saved: {OUTDIR / "v5_anisotropy_index.json"}')
