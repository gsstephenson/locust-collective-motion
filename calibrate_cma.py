"""
CMA-ES calibration of the hybrid v4 model against Weinburd 2024 field data.

Optimizes 6 parameters jointly:
    eta_base, lambda_pull, alpha_aniso,
    mu_responsiveness, k_pref, r_pref
against a weighted relative-error loss across three field metrics
(polarization, turning-angle std, NND median).

All params are rescaled to a [0, 10] internal space so a single sigma0
has uniform meaning across dimensions, per the pycma documentation.

Population members within each generation are evaluated in parallel
across CPU cores via multiprocessing.Pool. BLAS threads are pinned to 1
in workers to avoid oversubscription.

Run:
    conda activate csci5423
    python3 calibrate_cma.py --smoke      # fast smoke test (~30 s)
    python3 calibrate_cma.py --full       # full parallel run (~10 min on 32 cores)
"""

# Pin BLAS to 1 thread BEFORE any numpy import, so workers inherit it.
import os
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '1')

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np

from hybrid_v4 import evaluate, FIELD

BASE   = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
OUTDIR = BASE / 'week5_cma'
OUTDIR.mkdir(exist_ok=True)

# ── Parameter space ───────────────────────────────────────────────────────────
# (name, low, high) — internal CMA-ES variable lives in [0, 10]
PARAM_SPACE = [
    ('eta_base',          0.1,  1.5),
    ('lambda_pull',       0.0,  1.0),
    ('alpha_aniso',       0.0,  1.0),
    ('mu_responsiveness', 0.1,  1.0),
    ('k_pref',            0.0,  0.7),
    ('r_pref',            1.6,  5.0),   # must exceed r_repulsion=1.5
]
DIM = len(PARAM_SPACE)


def decode(x_internal):
    """Map a [0,10]^D vector to a params dict in physical units."""
    out = {}
    for xi, (name, lo, hi) in zip(x_internal, PARAM_SPACE):
        # Clip to [0,10] in case bounds are slightly violated
        xi_clip = max(0.0, min(10.0, float(xi)))
        out[name] = lo + (hi - lo) * xi_clip / 10.0
    return out


# ── Loss function ─────────────────────────────────────────────────────────────
# Weights reflect priority: polarization is already close in v3, the failures
# we need to close are turning-angle std and NND.
LOSS_WEIGHTS = dict(
    polarization = 1.0,
    turning_angle = 2.0,
    nnd = 2.0,
)


def loss_from_metrics(m):
    pol_err = abs(m['polarization_mean']  - FIELD['polarization_mean'])  / FIELD['polarization_mean']
    ta_err  = abs(m['turning_angle_std']  - FIELD['turning_angle_std'])  / FIELD['turning_angle_std']
    nnd_err = abs(m['nnd_median']         - FIELD['nnd_median'])         / FIELD['nnd_median']
    return (LOSS_WEIGHTS['polarization']  * pol_err
            + LOSS_WEIGHTS['turning_angle'] * ta_err
            + LOSS_WEIGHTS['nnd']           * nnd_err)


# ── Worker function (top-level so it pickles for Pool) ────────────────────────
def _eval_candidate(args):
    x, n_seeds = args
    params  = decode(x)
    metrics = evaluate(params, n_seeds=n_seeds)
    L       = loss_from_metrics(metrics)
    return params, metrics, L


# ── CMA-ES driver ─────────────────────────────────────────────────────────────
def run_cma(popsize, sigma0, max_gens, n_seeds, run_name, n_workers):
    log_path = OUTDIR / f'{run_name}_log.csv'
    fields = ['gen', 'eval', 'loss',
              *[p[0] for p in PARAM_SPACE],
              'polarization_mean', 'turning_angle_std', 'nnd_median']
    log_f = open(log_path, 'w', newline='')
    log_w = csv.writer(log_f)
    log_w.writerow(fields)

    es = cma.CMAEvolutionStrategy(
        x0     = [5.0] * DIM,                          # midpoint of [0,10]
        sigma0 = sigma0,
        inopts = dict(
            popsize     = popsize,
            bounds      = [[0.0]*DIM, [10.0]*DIM],
            verbose     = -9,
            tolx        = 1e-3,
            maxiter     = max_gens,
            seed        = 1,
        ),
    )

    eval_count = 0
    best = {'loss': float('inf'), 'params': None, 'metrics': None}
    t_start = time.time()

    print(f'CMA-ES: dim={DIM}, popsize={popsize}, sigma0={sigma0}, '
          f'max_gens={max_gens}, n_seeds={n_seeds}, n_workers={n_workers}')

    with Pool(processes=n_workers) as pool:
        for gen in range(max_gens):
            xs = es.ask()
            results = pool.map(_eval_candidate, [(x, n_seeds) for x in xs])
            losses = []
            for params, metrics, L in results:
                losses.append(L)
                eval_count += 1
                log_w.writerow([gen, eval_count, L,
                                *[params[p[0]] for p in PARAM_SPACE],
                                metrics['polarization_mean'],
                                metrics['turning_angle_std'],
                                metrics['nnd_median']])
                if L < best['loss']:
                    best = {'loss': L, 'params': dict(params),
                            'metrics': dict(metrics)}
            log_f.flush()
            es.tell(xs, losses)
            elapsed = time.time() - t_start
            print(f'  gen {gen:2d}  best={best["loss"]:.3f}  '
                  f'gen-mean={np.mean(losses):.3f}  '
                  f'gen-min={min(losses):.3f}  ({elapsed:.0f}s)',
                  flush=True)

    log_f.close()

    out = {
        'run_name':    run_name,
        'popsize':     popsize,
        'sigma0':      sigma0,
        'max_gens':    max_gens,
        'n_seeds':     n_seeds,
        'n_evals':     eval_count,
        'elapsed_sec': time.time() - t_start,
        'best': best,
        'field': FIELD,
        'loss_weights': LOSS_WEIGHTS,
    }
    out_json = OUTDIR / f'{run_name}_best.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nLog:  {log_path}')
    print(f'Best: {out_json}')
    print(f'\nBest loss: {best["loss"]:.3f}')
    print(f'Best params: {best["params"]}')
    print(f'Best metrics: {best["metrics"]}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke',              action='store_true', help='fast wiring test')
    g.add_argument('--full',               action='store_true', help='full optimization run')
    g.add_argument('--full-equal-weights', action='store_true',
                   help='full run with wPhi=wsigma=wNND=1 (sensitivity check)')
    args = ap.parse_args()

    n_cores = os.cpu_count() or 1
    if args.smoke:
        run_cma(popsize=8, sigma0=2.5, max_gens=3, n_seeds=2,
                run_name='smoke', n_workers=min(n_cores, 8))
    elif args.full_equal_weights:
        LOSS_WEIGHTS = dict(polarization=1.0, turning_angle=1.0, nnd=1.0)
        run_cma(popsize=20, sigma0=2.0, max_gens=40, n_seeds=5,
                run_name='equalw', n_workers=min(n_cores, 24))
    else:
        # popsize=20 fits the i9-13900K's effective parallelism (8 P-cores
        # + 16 E-cores) while staying in CMA-ES's productive popsize range
        # for a 6-D noisy objective.
        run_cma(popsize=20, sigma0=2.0, max_gens=40, n_seeds=5,
                run_name='full', n_workers=min(n_cores, 24))
