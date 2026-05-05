"""
Hybrid v4 — Anisotropic Vicsek + heading inertia + preferred spacing.

Extends hybrid v3 with two new mechanisms motivated by the slide-11 failure
analysis:

  (1) Heading inertia (mu_responsiveness in (0, 1]):
      Agents do not snap to the alignment-derived heading each tick; instead
      they partially commit to it via exponential smoothing in unit-vector
      space. mu=1 reproduces v3 exactly. Lower mu narrows the turning-angle
      distribution — directly attacks the 0.276-rad field target that v3
      misses by 215%.

  (2) Preferred-spacing repulsion (r_pref, k_pref):
      Soft radial push between r_repulsion and r_pref, with strength k_pref.
      Decouples NND from the alpha-aniso forward-weighting that currently
      compresses spacing as a side effect of producing the void.

The function `run_hybrid_v4(...)` returns (pos_history, heading_history).
The function `evaluate(params, n_seeds, ...)` returns an ensemble-mean
metrics dict — this is the interface CMA-ES will call.

Run directly to see a smoke test that v4 reproduces v3 when both new
mechanisms are disabled.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

BASE = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')

V3_DEFAULTS = dict(
    n_agents        = 150,
    Lx              = 99.44857450176137,
    Ly              = 55.93982315724077,
    v_min           = 2.7,
    v_max           = 11.76,
    eta_base        = 1.1,
    lambda_pull     = 0.8,
    alpha_aniso     = 0.6,
    cone_half_angle = np.pi / 3,
    r_interaction   = 7.0,
    r_repulsion     = 1.5,
    dt              = 0.04,
    n_steps         = 2000,
    # v4 additions
    mu_responsiveness = 1.0,   # 1.0 = no inertia (matches v3)
    r_pref            = 1.5,   # = r_repulsion → no soft zone (matches v3)
    k_pref            = 0.0,   # 0.0 = preferred-spacing disabled (matches v3)
    boundary          = 'periodic',  # 'periodic' or 'reflective'
)

FIELD = dict(
    polarization_mean = 0.8196,
    polarization_std  = 0.0615,
    turning_angle_std = 0.276,
    nnd_median        = 3.893,
    nnd_mean          = 4.473,
)


def run_hybrid_v4(*, n_agents, Lx, Ly, v_min, v_max,
                  eta_base, lambda_pull, alpha_aniso, cone_half_angle,
                  r_interaction, r_repulsion,
                  mu_responsiveness, r_pref, k_pref,
                  dt, n_steps, seed=42, boundary='periodic'):
    rng     = np.random.default_rng(seed)
    pos     = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    heading = rng.uniform(-np.pi, np.pi, size=n_agents)

    pos_history     = np.empty((n_steps + 1, n_agents, 2))
    heading_history = np.empty((n_steps + 1, n_agents))
    pos_history[0]     = pos
    heading_history[0] = heading

    for step in range(n_steps):
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        if boundary == 'periodic':
            delta[:, :, 0] -= Lx * np.round(delta[:, :, 0] / Lx)
            delta[:, :, 1] -= Ly * np.round(delta[:, :, 1] / Ly)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])

        in_range = (dist < r_interaction) & (dist > 0)

        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = bearing_abs - heading[:, np.newaxis]
        rel_bearing = (rel_bearing + np.pi) % (2 * np.pi) - np.pi

        in_cone = np.abs(rel_bearing) < cone_half_angle
        weights = np.where(in_cone, 1.0 + alpha_aniso, 1.0 - alpha_aniso)
        weights = weights * in_range.astype(float)
        weights = np.maximum(weights, 0)

        unit_vecs    = np.exp(1j * heading)
        weighted_sum = (weights * unit_vecs[np.newaxis, :]).sum(axis=1)
        avg_heading  = np.angle(weighted_sum)

        f_forward = (in_range & in_cone).sum(axis=1) / np.maximum(in_range.sum(axis=1), 1)
        eta_eff   = eta_base * (1.0 - lambda_pull * f_forward)
        noise     = rng.uniform(-eta_eff / 2, eta_eff / 2)

        desired_heading = avg_heading + noise

        # Hard repulsion override (unchanged from v3)
        rep_mask = (dist < r_repulsion) & (dist > 0)
        has_rep  = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            rep_heading = np.arctan2(rep_dy, rep_dx)
            desired_heading[has_rep] = rep_heading[has_rep] + noise[has_rep]

        # NEW: soft preferred-spacing zone — bias heading away from neighbors
        # in the band (r_repulsion, r_pref) by mixing in a radial push direction.
        # k_pref ∈ [0, 1] is the mixing weight; k_pref=0 disables the term.
        if k_pref > 0.0 and r_pref > r_repulsion:
            soft_mask = (dist > r_repulsion) & (dist < r_pref)
            has_soft  = soft_mask.any(axis=1)
            if has_soft.any():
                soft_dx = -(soft_mask * delta[:, :, 0]).sum(axis=1)
                soft_dy = -(soft_mask * delta[:, :, 1]).sum(axis=1)
                soft_heading = np.arctan2(soft_dy, soft_dx)
                desired_unit_align = np.exp(1j * desired_heading[has_soft])
                desired_unit_soft  = np.exp(1j * soft_heading[has_soft])
                mixed = ((1.0 - k_pref) * desired_unit_align
                         + k_pref       * desired_unit_soft)
                desired_heading[has_soft] = np.angle(mixed)

        # NEW: heading inertia via exponential smoothing in unit-vector space
        if mu_responsiveness >= 1.0:
            heading = desired_heading
        else:
            desired_unit = np.exp(1j * desired_heading)
            old_unit     = np.exp(1j * heading)
            mixed        = ((1.0 - mu_responsiveness) * old_unit
                            + mu_responsiveness       * desired_unit)
            heading      = np.angle(mixed)

        # Order-dependent speed (unchanged from v3)
        local_order = np.abs((in_range * unit_vecs[np.newaxis, :]).sum(axis=1) /
                             np.maximum(in_range.sum(axis=1), 1))
        speed = v_min + (v_max - v_min) * local_order

        new_x = pos[:, 0] + speed * dt * np.cos(heading)
        new_y = pos[:, 1] + speed * dt * np.sin(heading)

        if boundary == 'periodic':
            pos[:, 0] = new_x % Lx
            pos[:, 1] = new_y % Ly
        elif boundary == 'reflective':
            # Reflect off walls: clamp position and flip heading component.
            hit_xlo = new_x < 0
            hit_xhi = new_x > Lx
            hit_ylo = new_y < 0
            hit_yhi = new_y > Ly
            new_x = np.where(hit_xlo, -new_x, new_x)
            new_x = np.where(hit_xhi, 2 * Lx - new_x, new_x)
            new_y = np.where(hit_ylo, -new_y, new_y)
            new_y = np.where(hit_yhi, 2 * Ly - new_y, new_y)
            heading = np.where(hit_xlo | hit_xhi, np.pi - heading, heading)
            heading = np.where(hit_ylo | hit_yhi, -heading,         heading)
            heading = (heading + np.pi) % (2 * np.pi) - np.pi
            pos[:, 0] = np.clip(new_x, 0.0, Lx)
            pos[:, 1] = np.clip(new_y, 0.0, Ly)
        else:
            raise ValueError(f"Unknown boundary={boundary!r}")

        pos_history[step + 1]     = pos
        heading_history[step + 1] = heading

    return pos_history, heading_history


def polarization_from_headings(hh):
    return np.abs(np.exp(1j * hh).mean(axis=1))


def turning_angle_std(hh):
    dtheta = np.diff(hh, axis=0)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return float(np.std(dtheta))


def nnd_stats(pos_history, subsample=20):
    nnds = []
    for t in range(0, len(pos_history), subsample):
        d = cdist(pos_history[t], pos_history[t])
        np.fill_diagonal(d, np.inf)
        nnds.extend(np.min(d, axis=1))
    arr = np.array(nnds)
    return float(np.median(arr)), float(np.mean(arr))


def evaluate(params, n_seeds=5, burn_in=500, verbose=False):
    """Run n_seeds and return ensemble-mean metrics. Used by CMA-ES."""
    full = {**V3_DEFAULTS, **params}
    pol_m, ta_m, nnd_med, nnd_mn = [], [], [], []
    for seed in range(n_seeds):
        ph, hh = run_hybrid_v4(**full, seed=seed)
        pol_t  = polarization_from_headings(hh)
        pol_m.append(float(pol_t[burn_in:].mean()))
        ta_m.append(turning_angle_std(hh[burn_in:]))
        med, mn = nnd_stats(ph[burn_in:])
        nnd_med.append(med)
        nnd_mn.append(mn)
        if verbose:
            print(f'  seed {seed}: pol={pol_m[-1]:.3f} '
                  f'ta_std={ta_m[-1]:.3f} nnd_med={nnd_med[-1]:.2f}')
    return {
        'polarization_mean': float(np.mean(pol_m)),
        'polarization_std_across_seeds': float(np.std(pol_m)),
        'turning_angle_std': float(np.mean(ta_m)),
        'nnd_median':        float(np.mean(nnd_med)),
        'nnd_mean':          float(np.mean(nnd_mn)),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    print('Hybrid v4 smoke test')
    print('=' * 60)

    # 1. v4 at default params (mu=1, k_pref=0) should match v3
    print('\n[1] v4 at v3-equivalent settings (mu=1.0, k_pref=0):')
    t0 = time.time()
    m_v3eq = evaluate({}, n_seeds=3, verbose=True)
    print(f'  ensemble: pol={m_v3eq["polarization_mean"]:.3f}  '
          f'ta_std={m_v3eq["turning_angle_std"]:.3f}  '
          f'nnd_med={m_v3eq["nnd_median"]:.2f}  '
          f'({time.time()-t0:.1f}s)')

    # 2. v4 with inertia only — turning-angle std should drop
    print('\n[2] v4 with heading inertia (mu=0.3, k_pref=0):')
    t0 = time.time()
    m_inertia = evaluate({'mu_responsiveness': 0.3}, n_seeds=3, verbose=True)
    print(f'  ensemble: pol={m_inertia["polarization_mean"]:.3f}  '
          f'ta_std={m_inertia["turning_angle_std"]:.3f}  '
          f'nnd_med={m_inertia["nnd_median"]:.2f}  '
          f'({time.time()-t0:.1f}s)')

    print('\n' + '=' * 60)
    print(f'{"Metric":<22} {"Field":>8} {"v3-eq":>8} {"+inertia":>10}')
    print('-' * 60)
    print(f'{"Polarization":<22} {FIELD["polarization_mean"]:>8.3f} '
          f'{m_v3eq["polarization_mean"]:>8.3f} '
          f'{m_inertia["polarization_mean"]:>10.3f}')
    print(f'{"Turning-angle std":<22} {FIELD["turning_angle_std"]:>8.3f} '
          f'{m_v3eq["turning_angle_std"]:>8.3f} '
          f'{m_inertia["turning_angle_std"]:>10.3f}')
    print(f'{"NND median":<22} {FIELD["nnd_median"]:>8.3f} '
          f'{m_v3eq["nnd_median"]:>8.3f} '
          f'{m_inertia["nnd_median"]:>10.3f}')

    drop = (m_v3eq['turning_angle_std'] - m_inertia['turning_angle_std']) \
           / m_v3eq['turning_angle_std'] * 100
    print(f'\nTurning-angle-std drop from inertia: {drop:.1f}%')
    if drop > 5:
        print('PASS — inertia narrows the turning distribution as designed.')
    else:
        print('NOTE — drop smaller than expected; check mu_responsiveness wiring.')
