"""
Pull model — ordered initialization collapse figure
CSCI-5423 Final Project — Asher Albrecht & George Stephenson

Generates a publication-quality figure showing Phi(t) collapsing
from perfect order (Phi=1.0) to near-random (Phi~0.037) for the
pull model. Computes and annotates the collapse timescale.

Output:
  week4/figures/w4_pull_collapse.png

Run:
    conda activate locust
    python3 pull_collapse_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
WEEK4   = BASE / 'week4'
FIGURES = WEEK4 / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
PARAMS = dict(
    n_agents        = 150,
    Lx              = 99.44857450176137,
    Ly              = 55.93982315724077,
    speed           = 11.76,
    eta             = 0.2,
    r_interaction   = 7.0,
    r_repulsion     = 1.5,
    cone_half_angle = np.pi / 2,
    pull_strength   = 1.0,
    dt              = 0.04,
    n_steps         = 2000,
)
BURN_IN     = 500
N_SEEDS     = 5
FIELD_POL   = 0.8196
RANDOM_POL  = 0.0372   # from multi-seed run

# ── Pull model ────────────────────────────────────────────────────────────────
def run_pull(n_agents, Lx, Ly, speed, eta, r_interaction, r_repulsion,
             cone_half_angle, pull_strength, dt, n_steps, seed=42,
             init_headings=None):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    heading = (np.array(init_headings, dtype=float).copy()
               if init_headings is not None
               else rng.uniform(-np.pi, np.pi, size=n_agents))

    heading_history = np.empty((n_steps + 1, n_agents))
    heading_history[0] = heading

    for step in range(n_steps):
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        delta[:, :, 0] -= Lx * np.round(delta[:, :, 0] / Lx)
        delta[:, :, 1] -= Ly * np.round(delta[:, :, 1] / Ly)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])

        in_range    = (dist < r_interaction) & (dist > 0)
        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = bearing_abs - heading[:, np.newaxis]
        rel_bearing = (rel_bearing + np.pi) % (2 * np.pi) - np.pi

        weights     = np.cos(rel_bearing / 2) ** 2 * in_range.astype(float)
        z_neighbors = weights * np.exp(1j * bearing_abs)
        z_sum       = z_neighbors.sum(axis=1)
        w_sum       = weights.sum(axis=1)
        has_neighbors = w_sum > 0.01

        new_heading = heading.copy()
        if has_neighbors.any():
            weighted_bearing = np.angle(z_sum)
            vis      = has_neighbors
            z_current = np.exp(1j * heading[vis])
            z_target  = np.exp(1j * weighted_bearing[vis])
            z_blend   = (1 - pull_strength) * z_current + pull_strength * z_target
            new_heading[vis] = np.angle(z_blend)

        noise = rng.uniform(-eta / 2, eta / 2, size=n_agents)
        new_heading += noise

        rep_mask = (dist < r_repulsion) & (dist > 0)
        has_rep  = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            rep_heading = np.arctan2(rep_dy, rep_dx)
            new_heading[has_rep] = rep_heading[has_rep] + noise[has_rep]

        heading = new_heading
        pos[:, 0] = (pos[:, 0] + speed * dt * np.cos(heading)) % Lx
        pos[:, 1] = (pos[:, 1] + speed * dt * np.sin(heading)) % Ly
        heading_history[step + 1] = heading

    return heading_history

def polarization_from_headings(hh):
    return np.abs(np.exp(1j * hh).mean(axis=1))

# ── Run ordered-init seeds ────────────────────────────────────────────────────
print(f"Running pull model from ordered init — {N_SEEDS} seeds...")
t_axis  = np.arange(PARAMS['n_steps'] + 1) * PARAMS['dt']
traces  = []

for seed in range(N_SEEDS):
    rng_init = np.random.default_rng(seed + 100)
    init_h   = rng_init.normal(0, 0.01, size=PARAMS['n_agents'])
    print(f"  Seed {seed} — init Φ={np.abs(np.exp(1j*init_h).mean()):.4f} ...",
          end=' ', flush=True)
    hh    = run_pull(**PARAMS, seed=seed + 100, init_headings=init_h)
    pol_t = polarization_from_headings(hh)
    traces.append(pol_t)
    print(f"final Φ={pol_t[BURN_IN:].mean():.4f}")

traces_arr  = np.array(traces)
mean_trace  = traces_arr.mean(axis=0)

# ── Compute collapse timescale ────────────────────────────────────────────────
# Half-collapse: time to reach midpoint between initial (1.0) and final (~0.037)
half_target = (1.0 + RANDOM_POL) / 2
collapse_idx = np.argmax(mean_trace < half_target)
t_half = t_axis[collapse_idx] if collapse_idx > 0 else float('nan')

# Time to reach within 10% of random steady state
near_random  = RANDOM_POL * 1.10
collapse_idx_full = np.argmax(mean_trace < near_random)
t_full = t_axis[collapse_idx_full] if collapse_idx_full > 0 else float('nan')

print(f"\nCollapse timescale:")
print(f"  Half-collapse (Φ < {half_target:.3f}): t = {t_half:.2f} s")
print(f"  Near-random   (Φ < {near_random:.3f}): t = {t_full:.2f} s")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Individual seed traces
for i, pol_t in enumerate(traces):
    ax.plot(t_axis, pol_t, color='darkorange', lw=0.9, alpha=0.45,
            label='Individual seeds' if i == 0 else None)

# Ensemble mean
ax.plot(t_axis, mean_trace, color='darkorange', lw=2.5,
        label=f'Ensemble mean (n={N_SEEDS})')

# Reference lines
ax.axhline(FIELD_POL, color='red', ls='--', lw=1.5,
           label=f'Field target  Φ = {FIELD_POL:.3f}')
ax.axhline(RANDOM_POL, color='gray', ls='--', lw=1.2,
           label=f'Random steady state  Φ = {RANDOM_POL:.3f}')
ax.axhline(1.0, color='black', ls=':', lw=0.8, alpha=0.4)

# Annotate half-collapse
if not np.isnan(t_half):
    ax.axvline(t_half, color='darkorange', ls=':', lw=1.2, alpha=0.7)
    ax.annotate(f'Half-collapse\nt = {t_half:.1f} s',
                xy=(t_half, half_target),
                xytext=(t_half + 3, half_target + 0.15),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
                fontsize=9, color='darkorange')

# Annotate near-random
if not np.isnan(t_full):
    ax.axvline(t_full, color='gray', ls=':', lw=1.2, alpha=0.7)
    ax.annotate(f'Near-random\nt = {t_full:.1f} s',
                xy=(t_full, near_random),
                xytext=(t_full + 3, near_random + 0.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                fontsize=9, color='gray')

# Collapse magnitude annotation
ax.annotate('',
            xy=(1, RANDOM_POL + 0.01),
            xytext=(1, 0.98),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(1.8, 0.5, f'ΔΦ = {1.0 - RANDOM_POL:.3f}\n({((1.0 - RANDOM_POL)/1.0*100):.1f}% collapse)',
        fontsize=9, va='center', color='black')

ax.set(xlabel='Time (s)', ylabel='Polarization Φ',
       ylim=(-0.02, 1.08), xlim=(t_axis[0], t_axis[-1]),
       title='Pull Model — Order Destruction from Perfectly Ordered Initial State\n'
             f'N={PARAMS["n_agents"]} agents, η={PARAMS["eta"]} rad, '
             f'pull_strength={PARAMS["pull_strength"]}, {N_SEEDS} seeds')
ax.legend(fontsize=9, loc='center right')
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = FIGURES / 'w4_pull_collapse.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved: {out}')
