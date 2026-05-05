"""
Multi-seed Vicsek validation + burn-in check
CSCI-5423 Final Project — Asher Albrecht

Runs the calibrated Vicsek model across N_SEEDS seeds and reports:
  - Mean ± std for all four scalar metrics
  - Φ(t) burn-in verification plot
  - Exports results to week2/week2_results_multiseed.json

Run from the shared project directory:
    conda activate locust
    python3 vicsek_multiseed.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
WEEK2    = BASE / 'week2'
WEEK4    = BASE / 'week4'
FIGURES  = WEEK4 / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Calibrated parameters (from week2_results.json) ──────────────────────────
PARAMS = dict(
    n_agents      = 150,
    Lx            = 99.44857450176137,
    Ly            = 55.93982315724077,
    speed         = 11.76,
    eta           = 1.3,
    r_interaction = 7.0,
    r_repulsion   = 1.5,
    dt            = 0.04,
    n_steps       = 2000,
)
BURN_IN   = 500
N_SEEDS   = 10
SEEDS     = list(range(N_SEEDS))

FIELD = dict(
    polarization_mean = 0.8196,
    polarization_std  = 0.0615,
    turning_angle_std = 0.276,
    nnd_median        = 3.893,
    nnd_mean          = 4.473,
)

# ── Model ─────────────────────────────────────────────────────────────────────
def run_vicsek(n_agents, Lx, Ly, speed, eta, r_interaction, r_repulsion,
               dt, n_steps, seed=42):
    rng     = np.random.default_rng(seed)
    pos     = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    heading = rng.uniform(-np.pi, np.pi, size=n_agents)

    pos_history     = np.empty((n_steps + 1, n_agents, 2))
    heading_history = np.empty((n_steps + 1, n_agents))
    pos_history[0]     = pos
    heading_history[0] = heading

    for step in range(n_steps):
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        delta[:, :, 0] -= Lx * np.round(delta[:, :, 0] / Lx)
        delta[:, :, 1] -= Ly * np.round(delta[:, :, 1] / Ly)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])

        align_mask    = dist < r_interaction
        unit_vecs     = np.exp(1j * heading)
        neighbour_sum = align_mask @ unit_vecs
        avg_heading   = np.angle(neighbour_sum)
        noise         = rng.uniform(-eta/2, eta/2, size=n_agents)
        new_heading   = avg_heading + noise

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

        pos_history[step + 1]     = pos
        heading_history[step + 1] = heading

    return pos_history, heading_history

# ── Metric helpers ────────────────────────────────────────────────────────────
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
print(f"Running Vicsek model — {N_SEEDS} seeds, {PARAMS['n_steps']} steps each")
print(f"Parameters: N={PARAMS['n_agents']}, eta={PARAMS['eta']}, "
      f"r_int={PARAMS['r_interaction']}, dt={PARAMS['dt']}")
print()

pol_means_all  = []
pol_stds_all   = []
ta_stds_all    = []
nnd_medians_all = []
nnd_means_all  = []
pol_traces     = []   # for burn-in plot

for seed in SEEDS:
    print(f"  Seed {seed:2d} / {N_SEEDS-1} ...", end=' ', flush=True)
    ph, hh = run_vicsek(**PARAMS, seed=seed)

    pol_t       = polarization_from_headings(hh)
    pol_steady  = pol_t[BURN_IN:]
    ta_std      = turning_angle_std(hh[BURN_IN:])
    nnd_med, nnd_mn = nnd_stats(ph[BURN_IN:], PARAMS['Lx'], PARAMS['Ly'])

    pol_means_all.append(pol_steady.mean())
    pol_stds_all.append(pol_steady.std())
    ta_stds_all.append(ta_std)
    nnd_medians_all.append(nnd_med)
    nnd_means_all.append(nnd_mn)
    pol_traces.append(pol_t)

    print(f"pol={pol_steady.mean():.3f}  ta_std={ta_std:.3f}  "
          f"nnd_med={nnd_med:.2f}")

# ── Ensemble summary ──────────────────────────────────────────────────────────
def summary(arr):
    a = np.array(arr)
    return float(a.mean()), float(a.std())

pol_m,  pol_s  = summary(pol_means_all)
ta_m,   ta_s   = summary(ta_stds_all)
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
print(f"N_seeds = {N_SEEDS}  |  burn_in = {BURN_IN} steps  |  "
      f"n_steps = {PARAMS['n_steps']}")

# ── Burn-in verification plot ─────────────────────────────────────────────────
t_axis = np.arange(PARAMS['n_steps'] + 1) * PARAMS['dt']

fig, ax = plt.subplots(figsize=(12, 5))
for i, pol_t in enumerate(pol_traces):
    ax.plot(t_axis, pol_t, color='steelblue', lw=0.7, alpha=0.4)

pol_arr    = np.array(pol_traces)
pol_env_mean = pol_arr.mean(axis=0)
pol_env_std  = pol_arr.std(axis=0)

ax.plot(t_axis, pol_env_mean, color='steelblue', lw=2,
        label=f'Ensemble mean (n={N_SEEDS})')
ax.fill_between(t_axis,
                pol_env_mean - pol_env_std,
                pol_env_mean + pol_env_std,
                color='steelblue', alpha=0.2, label='±1 std across seeds')

ax.axvline(BURN_IN * PARAMS['dt'], color='gray', ls=':', lw=1.5,
           label=f'Burn-in cutoff ({BURN_IN} steps = {BURN_IN*PARAMS["dt"]:.0f}s)')
ax.axhline(FIELD['polarization_mean'], color='red', ls='--', lw=1.5,
           label=f'Field mean = {FIELD["polarization_mean"]:.3f}')
ax.axhline(FIELD['polarization_mean'] + FIELD['polarization_std'],
           color='red', ls=':', lw=0.8, alpha=0.5)
ax.axhline(FIELD['polarization_mean'] - FIELD['polarization_std'],
           color='red', ls=':', lw=0.8, alpha=0.5, label='Field ±1 std')

ax.set(xlabel='Time (s)', ylabel='Polarization Φ', ylim=(0, 1.05),
       title=f'Vicsek Model — Burn-in Verification ({N_SEEDS} seeds)\n'
             f'N={PARAMS["n_agents"]}, η={PARAMS["eta"]}, '
             f'r_int={PARAMS["r_interaction"]} cm')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

burnin_fig = FIGURES / 'w4_vicsek_burnin_multiseed.png'
fig.savefig(burnin_fig, dpi=150)
plt.show()
print(f'\nSaved burn-in figure: {burnin_fig}')

# ── Export JSON ───────────────────────────────────────────────────────────────
multiseed_results = {
    'model': 'vicsek',
    'n_seeds': N_SEEDS,
    'seeds': SEEDS,
    'params': PARAMS,
    'burn_in': BURN_IN,
    'per_seed': {
        'polarization_mean': pol_means_all,
        'polarization_std':  pol_stds_all,
        'turning_angle_std': ta_stds_all,
        'nnd_median':        nnd_medians_all,
        'nnd_mean':          nnd_means_all,
    },
    'ensemble': {
        'polarization_mean':      {'mean': pol_m,      'std': pol_s},
        'turning_angle_std':      {'mean': ta_m,       'std': ta_s},
        'nnd_median':             {'mean': nndmed_m,   'std': nndmed_s},
        'nnd_mean':               {'mean': nndmn_m,    'std': nndmn_s},
    },
    'field_targets': FIELD,
}

out_json = WEEK2 / 'week2_results_multiseed.json'
with open(out_json, 'w') as f:
    json.dump(multiseed_results, f, indent=2)
print(f'Saved: {out_json}')
