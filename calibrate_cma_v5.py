"""
CMA-ES calibration of the hybrid v5 model — adds rho_temporal as a 7th
dimension over the v4 optimization. Same parallelism, same loss, same
output format.

Run:
    conda activate csci5423
    python3 calibrate_cma_v5.py --smoke
    python3 calibrate_cma_v5.py --full
"""

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

from hybrid_v5 import evaluate, FIELD

BASE   = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
OUTDIR = BASE / 'week5_cma'
OUTDIR.mkdir(exist_ok=True)

# 7-D parameter space — v4's six dims + rho_temporal
PARAM_SPACE = [
    ('eta_base',          0.1,  1.5),
    ('lambda_pull',       0.0,  1.0),
    ('alpha_aniso',       0.0,  1.0),
    ('mu_responsiveness', 0.1,  1.0),
    ('k_pref',            0.0,  0.7),
    ('r_pref',            1.6,  5.0),
    ('rho_temporal',      0.05, 1.0),   # 0.05 = strong integration, 1.0 = none
]
DIM = len(PARAM_SPACE)


def decode(x_internal):
    out = {}
    for xi, (name, lo, hi) in zip(x_internal, PARAM_SPACE):
        xi_clip = max(0.0, min(10.0, float(xi)))
        out[name] = lo + (hi - lo) * xi_clip / 10.0
    return out


LOSS_WEIGHTS = dict(polarization=1.0, turning_angle=2.0, nnd=2.0)


def loss_from_metrics(m):
    pol_err = abs(m['polarization_mean']  - FIELD['polarization_mean'])  / FIELD['polarization_mean']
    ta_err  = abs(m['turning_angle_std']  - FIELD['turning_angle_std'])  / FIELD['turning_angle_std']
    nnd_err = abs(m['nnd_median']         - FIELD['nnd_median'])         / FIELD['nnd_median']
    return (LOSS_WEIGHTS['polarization']  * pol_err
            + LOSS_WEIGHTS['turning_angle'] * ta_err
            + LOSS_WEIGHTS['nnd']           * nnd_err)


def _eval_candidate(args):
    x, n_seeds = args
    params  = decode(x)
    metrics = evaluate(params, n_seeds=n_seeds)
    L       = loss_from_metrics(metrics)
    return params, metrics, L


def run_cma(popsize, sigma0, max_gens, n_seeds, run_name, n_workers):
    log_path = OUTDIR / f'{run_name}_log.csv'
    fields = ['gen', 'eval', 'loss',
              *[p[0] for p in PARAM_SPACE],
              'polarization_mean', 'turning_angle_std', 'nnd_median']
    log_f = open(log_path, 'w', newline='')
    log_w = csv.writer(log_f)
    log_w.writerow(fields)

    es = cma.CMAEvolutionStrategy(
        x0     = [5.0] * DIM,
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

    print(f'CMA-ES v5: dim={DIM}, popsize={popsize}, sigma0={sigma0}, '
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
        'model':       'hybrid_v5',
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
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--full',  action='store_true')
    args = ap.parse_args()

    n_cores = os.cpu_count() or 1
    if args.smoke:
        run_cma(popsize=8, sigma0=2.5, max_gens=3, n_seeds=2,
                run_name='v5_smoke', n_workers=min(n_cores, 8))
    else:
        # 7-D needs slightly more generations than 6-D for the same
        # convergence quality. popsize=20 still fills the cores.
        run_cma(popsize=20, sigma0=2.0, max_gens=50, n_seeds=5,
                run_name='v5_full', n_workers=min(n_cores, 24))
