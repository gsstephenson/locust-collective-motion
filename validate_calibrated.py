"""
Post-CMA-ES validation harness.

Run AFTER calibrate_cma.py --full finishes. Loads the best parameters from
week5_cma/full_best.json and:

  (a) Re-evaluates with a larger seed budget (n=20) to filter out
      lucky-seed artifacts from the smaller in-loop seed count.

  (b) Re-evaluates under reflective boundaries to test whether the
      calibrated dynamics depend on the periodic image structure
      (slide-11 idea #4: hopper bands are open systems).

  (c) Re-evaluates under v5 with temporal bearing integration ramping
      from rho=1.0 (v4-equivalent) to rho=0.05 to test whether adding
      Sayin-style integration buys anything beyond the v4 mechanisms
      (slide-11 idea #3).

Output: week5_cma/validation_report.json + console scorecard.
"""

import json
import time
from pathlib import Path

import numpy as np

from hybrid_v4 import evaluate as eval_v4, FIELD
from hybrid_v5 import evaluate as eval_v5

BASE   = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
OUTDIR = BASE / 'week5_cma'
BEST   = OUTDIR / 'full_best.json'


def relerr(obs, tgt):
    return abs(obs - tgt) / abs(tgt)


def scorecard(label, m):
    pol  = m['polarization_mean']
    ta   = m['turning_angle_std']
    nnd  = m['nnd_median']
    print(f'{label:<32} '
          f'pol={pol:.3f} ({100*relerr(pol, FIELD["polarization_mean"]):4.1f}%)  '
          f'ta_std={ta:.3f} ({100*relerr(ta, FIELD["turning_angle_std"]):5.1f}%)  '
          f'nnd_med={nnd:.2f} ({100*relerr(nnd, FIELD["nnd_median"]):4.1f}%)')


def main():
    if not BEST.exists():
        raise SystemExit(f'No best.json yet at {BEST}. '
                         f'Run calibrate_cma.py --full first.')

    with open(BEST) as f:
        best = json.load(f)

    p_star = best['best']['params']
    print(f'Loaded best params from {BEST}')
    for k, v in p_star.items():
        print(f'  {k:<22} = {v:.4f}')
    print(f'\nIn-loop loss: {best["best"]["loss"]:.3f}')
    print(f'In-loop metrics: {best["best"]["metrics"]}\n')

    print('=' * 100)
    print(f'{"Scenario":<32} {"polarization":>15} {"turning":>15} {"NND":>15}')
    print(f'{"":<32} {"(target 0.820)":>15} {"(target 0.276)":>15} {"(target 3.89)":>15}')
    print('=' * 100)

    # ── (a) Larger-seed re-evaluation ─────────────────────────────────────────
    t0 = time.time()
    m_largeN = eval_v4(p_star, n_seeds=20)
    scorecard('v4 / periodic / 20 seeds', m_largeN)
    print(f'  ({time.time()-t0:.0f}s)')

    # ── (b) Reflective-boundary stress test ───────────────────────────────────
    t0 = time.time()
    p_reflect = {**p_star, 'boundary': 'reflective'}
    m_reflect = eval_v4(p_reflect, n_seeds=10)
    scorecard('v4 / reflective / 10 seeds', m_reflect)
    print(f'  ({time.time()-t0:.0f}s)')

    # ── (c) v5 temporal-integration sweep ─────────────────────────────────────
    rho_vals = [1.0, 0.5, 0.2, 0.1, 0.05]
    v5_results = {}
    for rho in rho_vals:
        t0 = time.time()
        p_v5 = {**p_star, 'rho_temporal': rho}
        m_v5 = eval_v5(p_v5, n_seeds=5)
        v5_results[rho] = m_v5
        scorecard(f'v5 / rho={rho:.2f} / 5 seeds', m_v5)
        print(f'  ({time.time()-t0:.0f}s)')

    print('=' * 100)

    out = {
        'p_star':       p_star,
        'in_loop':      best['best']['metrics'],
        'large_seed':   m_largeN,
        'reflective':   m_reflect,
        'v5_rho_sweep': {f'{r:.2f}': m for r, m in v5_results.items()},
        'field':        FIELD,
    }
    out_path = OUTDIR / 'validation_report.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nReport: {out_path}')


if __name__ == '__main__':
    main()
