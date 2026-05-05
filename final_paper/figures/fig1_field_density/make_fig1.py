"""Figure 1: Field-data neighbor density map (Weinburd 2024).

Renders the body-centered density distribution from 6.8M neighbor counts
across 6 hopper-band clips, with the focal heading pointing up.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE     = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _palette import set_rcparams, SEQUENTIAL_CMAP
set_rcparams()

PROJECT  = HERE.parent.parent.parent
DATA_NPY = PROJECT / 'week5_cma' / 'field_density.npy'

density = np.load(DATA_NPY)
density_norm = density / density.sum()

fig, ax = plt.subplots(figsize=(5.5, 5))
edges = np.linspace(-10, 10, 51)
im = ax.imshow(density_norm.T, origin='lower',
               extent=[edges[0], edges[-1], edges[0], edges[-1]],
               cmap=SEQUENTIAL_CMAP, aspect='equal')
ax.plot(0, 0, 'w^', markersize=12)
ax.axhline(0, color='white', ls='--', alpha=0.5, lw=0.8)
ax.axvline(0, color='white', ls='--', alpha=0.5, lw=0.8)
ax.set_xlabel('Left to right (cm)')
ax.set_ylabel('Behind to ahead (cm)')
ax.set_title('Weinburd 2024 field data: body-centered neighbor density\n'
             '(walking + hopping locusts, 6 clips, 6.8M neighbor counts)')
plt.colorbar(im, ax=ax, label='Relative density')
plt.tight_layout()

out = HERE / 'fig1_field_density.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
