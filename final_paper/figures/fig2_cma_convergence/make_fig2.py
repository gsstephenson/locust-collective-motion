"""Figure 2: CMA-ES convergence — best-so-far loss across generations
for the v4 (6-D) and v5 (7-D) calibration runs."""

import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE    = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _palette import MODEL_COLORS, set_rcparams
set_rcparams()

PROJECT = HERE.parent.parent.parent
LOG_V4  = PROJECT / 'week5_cma' / 'full_log.csv'
LOG_V5  = PROJECT / 'week5_cma' / 'v5_full_log.csv'


def load_curve(path):
    gens, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row['gen']))
            losses.append(float(row['loss']))
    gens, losses = np.array(gens), np.array(losses)
    n_gens = gens.max() + 1
    best_so_far = np.empty(n_gens)
    gen_min     = np.empty(n_gens)
    gen_mean    = np.empty(n_gens)
    running_best = np.inf
    for g in range(n_gens):
        sel = losses[gens == g]
        gen_min[g]  = sel.min()
        gen_mean[g] = sel.mean()
        running_best = min(running_best, sel.min())
        best_so_far[g] = running_best
    return np.arange(n_gens), best_so_far, gen_min, gen_mean


g4, best4, gmin4, gmean4 = load_curve(LOG_V4)
g5, best5, gmin5, gmean5 = load_curve(LOG_V5)

c_v4 = MODEL_COLORS['Hybrid v4']
c_v5 = MODEL_COLORS['Hybrid v5']

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(g4, gmean4, 'o-', color=c_v4, alpha=0.35, lw=0.8, ms=3,
        label='v4 generation mean')
ax.plot(g4, best4, '-', color=c_v4, lw=2.0,
        label='v4 best so far (6-D)')
ax.plot(g5, gmean5, 's-', color=c_v5, alpha=0.35, lw=0.8, ms=3,
        label='v5 generation mean')
ax.plot(g5, best5, '-', color=c_v5, lw=2.0,
        label='v5 best so far (7-D)')
ax.set_xlabel('Generation')
ax.set_ylabel('Weighted loss')
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax.set_title('CMA-ES convergence: hybrid v4 (6-D) vs v5 (7-D)')

# Stagger annotations vertically so neither collides with the x-axis
# tick labels and they don't overlap each other.
ax.annotate(f'v4 final = {best4[-1]:.3f}',
            xy=(g4[-1], best4[-1]), xytext=(g4[-1]-14, best4[-1]*2.6),
            fontsize=9, color=c_v4,
            arrowprops=dict(arrowstyle='->', color=c_v4, lw=0.9))
ax.annotate(f'v5 final = {best5[-1]:.3f}',
            xy=(g5[-1], best5[-1]), xytext=(g5[-1]-18, best5[-1]*4.0),
            fontsize=9, color=c_v5,
            arrowprops=dict(arrowstyle='->', color=c_v5, lw=0.9))

plt.tight_layout()
out = HERE / 'fig2_cma_convergence.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
