"""
Hybrid v5 — v4 + temporal bearing integration (Sayin-analog ring-attractor).

Where v4 introduces *output* heading inertia (low-pass filter on the agent's
motion via mu_responsiveness), v5 introduces *input* temporal integration:
the alignment target itself is an exponential moving average of recent
neighbor-derived directions. This is the slide-11 idea #3 — Sayin et al.'s
ring attractor integrates bearings over time, which an instantaneous pull
or alignment rule cannot do.

Mechanism:
    Each agent maintains a complex unit-vector EMA of the alignment target:
        T_ema  ←  (1 - rho) * T_ema  +  rho * T_instantaneous
    The desired heading is angle(T_ema) instead of angle(T_instantaneous).

    rho_temporal ∈ (0, 1]:
        rho = 1.0  → no temporal integration (matches v4)
        rho = 0.1  → ~10-step memory window
        rho = 0.02 → ~50-step (~2 s at dt=0.04) memory window

State cost: one complex number per agent — negligible.

This is *distinct* from heading inertia (mu_responsiveness):
    - mu_responsiveness smooths the *agent's heading* between desired and
      previous heading (how committed the agent is to its motion).
    - rho_temporal smooths the *alignment signal* across time (how much the
      agent treats neighbor cues as a stable estimate vs. noisy snapshot).
A well-calibrated swarm probably needs both, but they can be turned on
independently.
"""

import numpy as np
from scipy.spatial.distance import cdist

V5_DEFAULTS = dict(
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
    mu_responsiveness = 1.0,
    r_pref            = 1.5,
    k_pref            = 0.0,
    boundary          = 'periodic',
    # v5 addition
    rho_temporal      = 1.0,   # 1.0 = no integration (matches v4)
)

FIELD = dict(
    polarization_mean = 0.8196,
    polarization_std  = 0.0615,
    turning_angle_std = 0.276,
    nnd_median        = 3.893,
    nnd_mean          = 4.473,
)


def run_hybrid_v5(*, n_agents, Lx, Ly, v_min, v_max,
                  eta_base, lambda_pull, alpha_aniso, cone_half_angle,
                  r_interaction, r_repulsion,
                  mu_responsiveness, r_pref, k_pref,
                  boundary, rho_temporal,
                  dt, n_steps, seed=42):
    rng     = np.random.default_rng(seed)
    pos     = rng.uniform([0, 0], [Lx, Ly], size=(n_agents, 2))
    heading = rng.uniform(-np.pi, np.pi, size=n_agents)
    target_ema = np.exp(1j * heading.copy())

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
        # Instantaneous target unit vector (preserves magnitude as confidence).
        # Fall back to current heading where there are no neighbors.
        no_neighbors = np.abs(weighted_sum) < 1e-9
        target_inst  = np.where(no_neighbors, unit_vecs, weighted_sum)
        # Normalize to unit magnitude before EMA (otherwise empty-neighborhood
        # steps would attenuate the EMA toward zero).
        target_inst = target_inst / np.maximum(np.abs(target_inst), 1e-12)

        # Temporal integration of alignment target
        if rho_temporal >= 1.0:
            target_ema = target_inst
        else:
            target_ema = (1.0 - rho_temporal) * target_ema + rho_temporal * target_inst
            # Re-normalize to keep it a unit vector
            target_ema = target_ema / np.maximum(np.abs(target_ema), 1e-12)

        avg_heading = np.angle(target_ema)

        f_forward = (in_range & in_cone).sum(axis=1) / np.maximum(in_range.sum(axis=1), 1)
        eta_eff   = eta_base * (1.0 - lambda_pull * f_forward)
        noise     = rng.uniform(-eta_eff / 2, eta_eff / 2)

        desired_heading = avg_heading + noise

        rep_mask = (dist < r_repulsion) & (dist > 0)
        has_rep  = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            rep_heading = np.arctan2(rep_dy, rep_dx)
            desired_heading[has_rep] = rep_heading[has_rep] + noise[has_rep]
            # Repulsion bypasses the temporal integrator on those agents
            target_ema[has_rep] = np.exp(1j * desired_heading[has_rep])

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

        if mu_responsiveness >= 1.0:
            heading = desired_heading
        else:
            desired_unit = np.exp(1j * desired_heading)
            old_unit     = np.exp(1j * heading)
            mixed = ((1.0 - mu_responsiveness) * old_unit
                     + mu_responsiveness       * desired_unit)
            heading = np.angle(mixed)

        local_order = np.abs((in_range * unit_vecs[np.newaxis, :]).sum(axis=1) /
                             np.maximum(in_range.sum(axis=1), 1))
        speed = v_min + (v_max - v_min) * local_order

        new_x = pos[:, 0] + speed * dt * np.cos(heading)
        new_y = pos[:, 1] + speed * dt * np.sin(heading)

        if boundary == 'periodic':
            pos[:, 0] = new_x % Lx
            pos[:, 1] = new_y % Ly
        elif boundary == 'reflective':
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
    full = {**V5_DEFAULTS, **params}
    pol_m, ta_m, nnd_med, nnd_mn = [], [], [], []
    for seed in range(n_seeds):
        ph, hh = run_hybrid_v5(**full, seed=seed)
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


if __name__ == '__main__':
    import time

    print('Hybrid v5 smoke test — temporal bearing integration')
    print('=' * 64)

    cases = [
        ('v5 at v4-equiv (rho=1)',                {'rho_temporal': 1.0}),
        ('v5 with temporal integ (rho=0.2)',      {'rho_temporal': 0.2}),
        ('v5 with strong integ (rho=0.05)',       {'rho_temporal': 0.05}),
    ]

    print(f'{"Case":<40} {"pol":>6} {"ta_std":>7} {"nnd_med":>8}')
    print('-' * 64)
    for label, params in cases:
        t0 = time.time()
        m = evaluate(params, n_seeds=3)
        print(f'{label:<40} {m["polarization_mean"]:>6.3f} '
              f'{m["turning_angle_std"]:>7.3f} {m["nnd_median"]:>8.2f}  '
              f'({time.time()-t0:.0f}s)')

    print(f'\nField target: pol={FIELD["polarization_mean"]:.3f} '
          f'ta_std={FIELD["turning_angle_std"]:.3f} '
          f'nnd_med={FIELD["nnd_median"]:.2f}')
