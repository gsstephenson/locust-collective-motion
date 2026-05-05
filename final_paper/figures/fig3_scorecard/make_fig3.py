"""Figure 3: Final scorecard — relative error of every model on each metric.

Shows polarization Phi, turning-angle std, NND median across:
Vicsek (calibrated), Pull (calibrated), Hybrid v3, Hybrid v4 calibrated,
Hybrid v5 calibrated.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

HERE    = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _palette import MODEL_COLORS, MODEL_HATCHES, set_rcparams
set_rcparams()

PROJECT = HERE.parent.parent.parent
W2 = PROJECT / 'week2'
W4 = PROJECT / 'week4'
W5 = PROJECT / 'week5_cma'

with open(W2 / 'week2_results.json')                  as f: vicsek = json.load(f)['metrics']
with open(W2 / 'week2_pull_results.json')             as f: pull   = json.load(f)['metrics']
with open(W2 / 'week2_hybrid_results_multiseed.json') as f: v3_ms  = json.load(f)
v3 = {k: v3_ms['ensemble'][k]['mean'] for k in
      ['polarization_mean', 'turning_angle_std', 'nnd_median']}

with open(W5 / 'full_best.json')    as f: v4 = json.load(f)['best']['metrics']
with open(W5 / 'v5_full_best.json') as f: v5 = json.load(f)['best']['metrics']
with open(W4 / 'week4_results.json') as f: ft = json.load(f)['field_targets']

models  = ['Vicsek', 'Pull', 'Hybrid v3', 'Hybrid v4', 'Hybrid v5']
colors  = [MODEL_COLORS[m]  for m in models]
hatches = [MODEL_HATCHES[m] for m in models]
metric_keys = ['polarization_mean', 'turning_angle_std', 'nnd_median']
labels = ['Polarization $\\Phi$', 'Turning-angle std (rad)',
          'NND median (cm)']
m_data = [vicsek, pull, v3, v4, v5]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
x = np.arange(len(models))

for ax, key, label in zip(axes, metric_keys, labels):
    vals = [d[key] for d in m_data]
    bars = ax.bar(x, vals, color=colors, alpha=0.9,
                  edgecolor='black', linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.axhline(ft[key], color='black', ls='--', lw=1.2, alpha=0.7,
               label=f'Field = {ft[key]:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_title(label, fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + ax.get_ylim()[1]*0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Per-metric comparison against Weinburd 2024 field targets',
             fontsize=12, y=1.02)
plt.tight_layout()

out = HERE / 'fig3_scorecard.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')

import csv as _csv
with open(HERE / 'fig3_scorecard.csv', 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['model', *metric_keys, 'pol_pct_err',
                'ta_std_pct_err', 'nnd_pct_err'])
    for m, d in zip(models, m_data):
        row = [m, d['polarization_mean'], d['turning_angle_std'], d['nnd_median'],
               100 * abs(d['polarization_mean'] - ft['polarization_mean']) / ft['polarization_mean'],
               100 * abs(d['turning_angle_std'] - ft['turning_angle_std']) / ft['turning_angle_std'],
               100 * abs(d['nnd_median']        - ft['nnd_median'])        / ft['nnd_median']]
        w.writerow(row)
print(f'Saved: {HERE / "fig3_scorecard.csv"}')
