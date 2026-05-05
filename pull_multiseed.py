"""
Multi-seed Pull model validation + ordered-initialization test
CSCI-5423 Final Project — Asher Albrecht & George Stephenson

Runs:
  1. Multi-seed Pull (10 seeds, random init) — confirms failure is not seed-dependent
  2. Ordered-initialization test — answers whether Pull maintains or destroys order

Output:
  week2/week2_pull_results_multiseed.json
  week4/figures/w4_pull_burnin_multiseed.png
  week4/figures/w4_pull_ordered_init.png

Run:
    conda activate locust
    python3 pull_multiseed.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
WEEK2   = BASE / 'week2'
WEEK4   = BASE / 'week4'
FIGURES = WEEK4 / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Parameters (from week2_pull_results.json) ─────────────────────────────────
PARAMS = dict(
    n_agents       = 150,
    Lx             = 99.44857450176137,
    Ly             = 55.93982315724077,
    speed          = 11.76,
    eta            = 0.2,
    r_interaction  = 7.0,
    r_repulsion    = 1.5,
    cone_half_angle= np.pi / 2,
    pull_strength  = 1.0,
    dt             = 0.04,
    n_steps        = 2000,
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

# ── Pull model (copied exactly from George's week2_pull_model.ipynb) ──────────
def run_pull(n_agents, Lx, Ly, speed, eta, r_interaction, r_repulsion,
             cone_half_angle, pull_strength, dt, n_steps, seed=42,
             init_headings=None):
    """
    Pull model with optional ordered initialization.
    init_headings: if provided (array of shape n_agents), use as initial headings.
                   if None, sample uniformly from [-pi, pi] (random init).
    """
    rng = np.random.default_rng(seed)

    pos = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    if init_headings is not None:
        heading = np.array(init_headings, dtype=float).copy()
    else:
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

        in_range    = (dist < r_interaction) & (dist > 0)
        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = bearing_abs - heading[:, np.newaxis]
        rel_bearing = (rel_bearing + np.pi) % (2 * np.pi) - np.pi

        weights    = np.cos(rel_bearing / 2) ** 2 * in_range.astype(float)
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

        pos_history[step + 1]     = pos
        heading_history[step + 1] = heading

    return pos_history, heading_history

def polarization_from_headings(hh):
    return np.abs(np.exp(1j * hh).mean(axis=1))

# ── Part 1: Multi-seed random init ───────────────────────────────────────────
print("=" * 60)
print("Part 1: Multi-seed Pull (random initialization)")
print(f"  {N_SEEDS} seeds, {PARAMS['n_steps']} steps, eta={PARAMS['eta']}")
print("=" * 60)

pol_traces = []
pol_means_all = []

for seed in SEEDS:
    print(f"  Seed {seed:2d} / {N_SEEDS-1} ...", end=' ', flush=True)
    _, hh   = run_pull(**PARAMS, seed=seed)
    pol_t   = polarization_from_headings(hh)
    pol_steady = pol_t[BURN_IN:]
    pol_means_all.append(pol_steady.mean())
    pol_traces.append(pol_t)
    print(f"pol={pol_steady.mean():.4f}")

pol_arr = np.array(pol_means_all)
print(f"\nEnsemble polarization: mean={pol_arr.mean():.4f}  std={pol_arr.std():.4f}")
print(f"Field target: {FIELD['polarization_mean']:.3f}")
print(f"Ratio to field: {pol_arr.mean() / FIELD['polarization_mean']:.3f}")

# Burn-in figure
t_axis   = np.arange(PARAMS['n_steps'] + 1) * PARAMS['dt']
traces   = np.array(pol_traces)
env_mean = traces.mean(axis=0)
env_std  = traces.std(axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
for pol_t in pol_traces:
    ax.plot(t_axis, pol_t, color='steelblue', lw=0.7, alpha=0.4)
ax.plot(t_axis, env_mean, color='steelblue', lw=2,
        label=f'Ensemble mean (n={N_SEEDS})')
ax.fill_between(t_axis, env_mean - env_std, env_mean + env_std,
                color='steelblue', alpha=0.2, label='±1 std across seeds')
ax.axvline(BURN_IN * PARAMS['dt'], color='gray', ls=':', lw=1.5,
           label=f'Burn-in cutoff (20s)')
ax.axhline(FIELD['polarization_mean'], color='red', ls='--', lw=1.5,
           label=f'Field target = {FIELD["polarization_mean"]:.3f}')
ax.set(xlabel='Time (s)', ylabel='Polarization Φ', ylim=(0, 1.05),
       title=f'Pull Model — Multi-seed Validation ({N_SEEDS} seeds, random init)\n'
             f'η={PARAMS["eta"]}, pull_strength={PARAMS["pull_strength"]}')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out1 = FIGURES / 'w4_pull_burnin_multiseed.png'
fig.savefig(out1, dpi=150)
plt.show()
print(f'Saved: {out1}')

# ── Part 2: Ordered initialization test ──────────────────────────────────────
print()
print("=" * 60)
print("Part 2: Ordered initialization test")
print("  Q: Does Pull MAINTAIN order when seeded from an ordered state?")
print("  Random init → already shown to fail (pol~0.06)")
print("  Ordered init → all headings ≈ 0 + tiny noise (std=0.01 rad)")
print("=" * 60)

N_ORDERED_SEEDS = 5
ordered_traces  = []
random_traces_subset = pol_traces[:N_ORDERED_SEEDS]

for seed in range(N_ORDERED_SEEDS):
    rng_init = np.random.default_rng(seed + 100)
    # Near-perfect order: all headings ≈ 0 with tiny noise
    init_h = rng_init.normal(0, 0.01, size=PARAMS['n_agents'])
    print(f"  Ordered seed {seed} — init pol={np.abs(np.exp(1j*init_h).mean()):.4f} ...",
          end=' ', flush=True)
    _, hh  = run_pull(**PARAMS, seed=seed + 100, init_headings=init_h)
    pol_t  = polarization_from_headings(hh)
    ordered_traces.append(pol_t)
    print(f"final pol={pol_t[BURN_IN:].mean():.4f}")

ordered_arr  = np.array(ordered_traces)
ordered_mean = ordered_arr.mean(axis=0)
ordered_std  = ordered_arr.std(axis=0)

random_arr   = np.array(random_traces_subset)
random_mean  = random_arr.mean(axis=0)
random_std   = random_arr.std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: ordered init
ax = axes[0]
for tr in ordered_traces:
    ax.plot(t_axis, tr, color='darkorange', lw=0.8, alpha=0.5)
ax.plot(t_axis, ordered_mean, color='darkorange', lw=2,
        label=f'Ordered init (n={N_ORDERED_SEEDS})')
ax.fill_between(t_axis, ordered_mean - ordered_std, ordered_mean + ordered_std,
                color='darkorange', alpha=0.2)
ax.axhline(FIELD['polarization_mean'], color='red', ls='--', lw=1.5,
           label='Field target')
ax.axvline(BURN_IN * PARAMS['dt'], color='gray', ls=':', lw=1)
ax.set(xlabel='Time (s)', ylabel='Polarization Φ', ylim=(0, 1.05),
       title='Pull — Ordered Initialization\n(all headings ≈ 0 + noise std=0.01)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: random init (same seeds for comparison)
ax2 = axes[1]
for tr in random_traces_subset:
    ax2.plot(t_axis, tr, color='steelblue', lw=0.8, alpha=0.5)
ax2.plot(t_axis, random_mean, color='steelblue', lw=2,
         label=f'Random init (n={N_ORDERED_SEEDS})')
ax2.fill_between(t_axis, random_mean - random_std, random_mean + random_std,
                 color='steelblue', alpha=0.2)
ax2.axhline(FIELD['polarization_mean'], color='red', ls='--', lw=1.5,
            label='Field target')
ax2.axvline(BURN_IN * PARAMS['dt'], color='gray', ls=':', lw=1)
ax2.set(xlabel='Time (s)', ylabel='Polarization Φ', ylim=(0, 1.05),
        title='Pull — Random Initialization\n(baseline failure)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Pull Model — Ordered vs Random Initialization\n'
             'Does pull maintain order, or does it destroy it?',
             fontsize=13)
plt.tight_layout()
out2 = FIGURES / 'w4_pull_ordered_init.png'
fig.savefig(out2, dpi=150)
plt.show()
print(f'Saved: {out2}')

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ORDERED INIT RESULT INTERPRETATION")
print("=" * 60)
ord_final = np.array([tr[BURN_IN:].mean() for tr in ordered_traces])
rnd_final = np.array([tr[BURN_IN:].mean() for tr in random_traces_subset])
print(f"Random init  — steady-state pol: {rnd_final.mean():.4f} ± {rnd_final.std():.4f}")
print(f"Ordered init — steady-state pol: {ord_final.mean():.4f} ± {ord_final.std():.4f}")
print()
if ord_final.mean() < 0.2:
    print("RESULT: Pull DESTROYS order — even from a perfectly ordered start,")
    print("        the model relaxes to near-random. The pull mechanism has no")
    print("        order-preserving dynamics whatsoever.")
elif ord_final.mean() < 0.6:
    print("RESULT: Pull PARTIALLY maintains order — some alignment signal,")
    print("        but decays substantially. Mechanism is weak.")
else:
    print("RESULT: Pull MAINTAINS order from ordered init but cannot generate")
    print("        it from disorder. Pure bootstrapping failure.")

# ── Export JSON ───────────────────────────────────────────────────────────────
results = {
    'model': 'pull',
    'n_seeds': N_SEEDS,
    'params': {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
               for k, v in PARAMS.items()},
    'burn_in': BURN_IN,
    'random_init': {
        'per_seed_pol_mean': pol_means_all,
        'ensemble_pol_mean': float(pol_arr.mean()),
        'ensemble_pol_std':  float(pol_arr.std()),
    },
    'ordered_init': {
        'per_seed_pol_mean': ord_final.tolist(),
        'ensemble_pol_mean': float(ord_final.mean()),
        'ensemble_pol_std':  float(ord_final.std()),
    },
    'field_targets': FIELD,
}

out_json = WEEK2 / 'week2_pull_results_multiseed.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_json}')
