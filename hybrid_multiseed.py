"""
Multi-seed Anisotropic Vicsek (Hybrid v3) validation
CSCI-5423 Final Project — Asher Albrecht & George Stephenson

Runs the calibrated hybrid model across 10 seeds and reports
ensemble mean ± std for all four scalar metrics.

Output:
  week2/week2_hybrid_results_multiseed.json
  week4/figures/w4_hybrid_burnin_multiseed.png

Run:
    conda activate locust
    python3 hybrid_multiseed.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
WEEK2   = BASE / 'week2'
WEEK4   = BASE / 'week4'
FIGURES = WEEK4 / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Parameters (from week2_hybrid_results.json) ───────────────────────────────
PARAMS = dict(
    n_agents        = 150,
    Lx              = 99.44857450176137,
    Ly              = 55.93982315724077,
    v_min           = 2.7,
    v_max           = 11.76,
    eta_base        = 1.1,
    lambda_pull     = 0.8,
    alpha_aniso     = 0.6,
    cone_half_angle = np.pi / 3,      # 60 degrees
    r_interaction   = 7.0,
    r_repulsion     = 1.5,
    dt              = 0.04,
    n_steps         = 2000,
)
BURN_IN = 500
N_SEEDS = 10
SEEDS   = list(range(N_SEEDS))

FIELD = dict(
    polarization_mean = 0.8196,
    polarization_std  = 0.0615,
    turning_angle_std = 0.276,
    nnd_median        = 3.893,
    nnd_mean          = 4.473,
)

# ── Hybrid model ──────────────────────────────────────────────────────────────
def run_hybrid(n_agents, Lx, Ly, v_min, v_max, eta_base, lambda_pull,
               alpha_aniso, cone_half_angle, r_interaction, r_repulsion,
               dt, n_steps, seed=42):
    """
    Anisotropic Vicsek (Hybrid v3):
      - Alignment weighting: neighbors ahead (within cone_half_angle) weighted
        by (1 + alpha_aniso), neighbors behind by (1 - alpha_aniso)
      - Anisotropic noise: eta_eff = eta_base * (1 - lambda_pull * f_forward)
      - Speed variation: speed scales with local order (v_min to v_max)
    """
    rng     = np.random.default_rng(seed)
    pos     = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    heading = rng.uniform(-np.pi, np.pi, size=n_agents)
    speed   = np.full(n_agents, (v_min + v_max) / 2)

    pos_history     = np.empty((n_steps + 1, n_agents, 2))
    heading_history = np.empty((n_steps + 1, n_agents))
    pos_history[0]     = pos
    heading_history[0] = heading

    for step in range(n_steps):
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        delta[:, :, 0] -= Lx * np.round(delta[:, :, 0] / Lx)
        delta[:, :, 1] -= Ly * np.round(delta[:, :, 1] / Ly)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])

        in_range    = (dist < r_interaction) & (dist > 0)

        # Relative bearing of each neighbor from focal agent's heading
        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = bearing_abs - heading[:, np.newaxis]
        rel_bearing = (rel_bearing + np.pi) % (2 * np.pi) - np.pi

        # Anisotropic alignment weights
        in_cone  = np.abs(rel_bearing) < cone_half_angle    # (N, N)
        weights  = np.where(in_cone, 1.0 + alpha_aniso, 1.0 - alpha_aniso)
        weights  = weights * in_range.astype(float)
        weights  = np.maximum(weights, 0)

        # Weighted circular mean of neighbor headings
        unit_vecs    = np.exp(1j * heading)                  # (N,)
        weighted_sum = (weights * unit_vecs[np.newaxis, :]).sum(axis=1)  # (N,)
        avg_heading  = np.angle(weighted_sum)

        # Anisotropic noise suppression
        f_forward = (in_range & in_cone).sum(axis=1) / np.maximum(in_range.sum(axis=1), 1)
        eta_eff   = eta_base * (1.0 - lambda_pull * f_forward)
        noise     = np.array([rng.uniform(-e/2, e/2) for e in eta_eff])

        new_heading = avg_heading + noise

        # Repulsion override
        rep_mask = (dist < r_repulsion) & (dist > 0)
        has_rep  = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            rep_heading = np.arctan2(rep_dy, rep_dx)
            new_heading[has_rep] = rep_heading[has_rep] + noise[has_rep]

        heading = new_heading

        # Order-dependent speed
        local_order = np.abs((in_range * unit_vecs[np.newaxis, :]).sum(axis=1) /
                             np.maximum(in_range.sum(axis=1), 1))
        speed = v_min + (v_max - v_min) * local_order

        pos[:, 0] = (pos[:, 0] + speed * dt * np.cos(heading)) % Lx
        pos[:, 1] = (pos[:, 1] + speed * dt * np.sin(heading)) % Ly

        pos_history[step + 1]     = pos
        heading_history[step + 1] = heading

    return pos_history, heading_history

def polarization_from_headings(hh):
    return np.abs(np.exp(1j * hh).mean(axis=1))

def turning_angle_std(hh):
    dtheta = np.diff(hh, axis=0)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return float(np.std(dtheta))

def nnd_stats(pos_history, Lx, Ly, subsample=20):
    nnds = []
    for t in range(0, len(pos_history), subsample):
        pos   = pos_history[t]
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        nnds.extend(np.min(dists, axis=1))
    arr = np.array(nnds)
    return float(np.median(arr)), float(np.mean(arr))

# ── Multi-seed loop ───────────────────────────────────────────────────────────
print(f"Running Anisotropic Vicsek (Hybrid v3) — {N_SEEDS} seeds")
print(f"Parameters: η_base={PARAMS['eta_base']}, λ={PARAMS['lambda_pull']}, "
      f"α={PARAMS['alpha_aniso']}, N={PARAMS['n_agents']}")
print()

pol_means_all   = []
pol_stds_all    = []
ta_stds_all     = []
nnd_medians_all = []
nnd_means_all   = []
pol_traces      = []

for seed in SEEDS:
    print(f"  Seed {seed:2d} / {N_SEEDS-1} ...", end=' ', flush=True)
    ph, hh     = run_hybrid(**PARAMS, seed=seed)
    pol_t      = polarization_from_headings(hh)
    pol_steady = pol_t[BURN_IN:]
    ta_std     = turning_angle_std(hh[BURN_IN:])
    nnd_med, nnd_mn = nnd_stats(ph[BURN_IN:], PARAMS['Lx'], PARAMS['Ly'])

    pol_means_all.append(pol_steady.mean())
    pol_stds_all.append(pol_steady.std())
    ta_stds_all.append(ta_std)
    nnd_medians_all.append(nnd_med)
    nnd_means_all.append(nnd_mn)
    pol_traces.append(pol_t)

    print(f"pol={pol_steady.mean():.3f}  ta_std={ta_std:.3f}  nnd_med={nnd_med:.2f}")

def summary(arr):
    a = np.array(arr)
    return float(a.mean()), float(a.std())

pol_m,    pol_s    = summary(pol_means_all)
ta_m,     ta_s     = summary(ta_stds_all)
nndmed_m, nndmed_s = summary(nnd_medians_all)
nndmn_m,  nndmn_s  = summary(nnd_means_all)

print()
print("=" * 65)
print(f"{'Metric':<30} {'Field':>10} {'Mean':>10} {'±Std':>8}")
print("=" * 65)
print(f"{'Polarization mean':<30} {FIELD['polarization_mean']:>10.3f} "
      f"{pol_m:>10.3f} {pol_s:>8.4f}")
print(f"{'Turning angle std (rad)':<30} {FIELD['turning_angle_std']:>10.3f} "
      f"{ta_m:>10.3f} {ta_s:>8.4f}")
print(f"{'NND median (cm)':<30} {FIELD['nnd_median']:>10.3f} "
      f"{nndmed_m:>10.3f} {nndmed_s:>8.4f}")
print(f"{'NND mean (cm)':<30} {FIELD['nnd_mean']:>10.3f} "
      f"{nndmn_m:>10.3f} {nndmn_s:>8.4f}")
print("=" * 65)

# ── Burn-in figure ────────────────────────────────────────────────────────────
t_axis   = np.arange(PARAMS['n_steps'] + 1) * PARAMS['dt']
traces   = np.array(pol_traces)
env_mean = traces.mean(axis=0)
env_std  = traces.std(axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
for pol_t in pol_traces:
    ax.plot(t_axis, pol_t, color='#27a060', lw=0.7, alpha=0.4)
ax.plot(t_axis, env_mean, color='#27a060', lw=2,
        label=f'Ensemble mean (n={N_SEEDS})')
ax.fill_between(t_axis, env_mean - env_std, env_mean + env_std,
                color='#27a060', alpha=0.2, label='±1 std across seeds')
ax.axvline(BURN_IN * PARAMS['dt'], color='gray', ls=':', lw=1.5,
           label=f'Burn-in cutoff (20s)')
ax.axhline(FIELD['polarization_mean'], color='red', ls='--', lw=1.5,
           label=f'Field target = {FIELD["polarization_mean"]:.3f}')
ax.axhline(FIELD['polarization_mean'] + FIELD['polarization_std'],
           color='red', ls=':', lw=0.8, alpha=0.5)
ax.axhline(FIELD['polarization_mean'] - FIELD['polarization_std'],
           color='red', ls=':', lw=0.8, alpha=0.5, label='Field ±1 std')
ax.set(xlabel='Time (s)', ylabel='Polarization Φ', ylim=(0, 1.05),
       title=f'Anisotropic Vicsek (Hybrid v3) — Burn-in Verification ({N_SEEDS} seeds)\n'
             f'η_base={PARAMS["eta_base"]}, λ={PARAMS["lambda_pull"]}, '
             f'α={PARAMS["alpha_aniso"]}, N={PARAMS["n_agents"]}')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_fig = FIGURES / 'w4_hybrid_burnin_multiseed.png'
fig.savefig(out_fig, dpi=150)
plt.show()
print(f'\nSaved: {out_fig}')

# ── Export JSON ───────────────────────────────────────────────────────────────
results = {
    'model': 'anisotropic_vicsek_v3',
    'n_seeds': N_SEEDS,
    'seeds': SEEDS,
    'params': {k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in PARAMS.items()},
    'burn_in': BURN_IN,
    'per_seed': {
        'polarization_mean': pol_means_all,
        'polarization_std':  pol_stds_all,
        'turning_angle_std': ta_stds_all,
        'nnd_median':        nnd_medians_all,
        'nnd_mean':          nnd_means_all,
    },
    'ensemble': {
        'polarization_mean': {'mean': pol_m,    'std': pol_s},
        'turning_angle_std': {'mean': ta_m,     'std': ta_s},
        'nnd_median':        {'mean': nndmed_m, 'std': nndmed_s},
        'nnd_mean':          {'mean': nndmn_m,  'std': nndmn_s},
    },
    'field_targets': FIELD,
}

out_json = WEEK2 / 'week2_hybrid_results_multiseed.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {out_json}')
