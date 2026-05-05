"""Figure 4: Spatial check — body-centered neighbor density at the v5 optimum
side by side with the Weinburd field reference and the v3 hybrid.

The aim is to show that the calibrated v5 model reproduces the (essentially
isotropic) field structure, contrary to the v3 figure that visually suggested
a forward void.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE    = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _palette import set_rcparams, SEQUENTIAL_CMAP
set_rcparams()

PROJECT = HERE.parent.parent.parent
W5      = PROJECT / 'week5_cma'

field = np.load(W5 / 'field_density.npy')
v3    = np.load(W5 / 'v3_metric3_density.npy')
v4    = np.load(W5 / 'v4_metric3_density.npy')
v5    = np.load(W5 / 'v5_metric3_density.npy')

# Pixel grid (50x50 over [-10, 10] cm). Axis 0 is x (left-right),
# axis 1 is y (behind-ahead, +y forward).
_edges   = np.linspace(-10, 10, 51)
_centers = (_edges[:-1] + _edges[1:]) / 2
XG, YG   = np.meshgrid(_centers, _centers, indexing='ij')

def aniso_global(d):
    cy = d.shape[1] // 2
    rear  = d[:, :cy].mean()
    front = d[:, cy:].mean()
    return (rear - front) / (rear + front)

# Restricted to a radial band centered on the focal. The global index
# averages over a 10 cm half-disk, which dilutes any near-zone signal
# from the preferred-spacing ring. Reporting both indices reveals the
# scale at which spatial structure differs.
R_NEAR_IN, R_NEAR_OUT = 1.5, 3.0

def aniso_near(d, r_in=R_NEAR_IN, r_out=R_NEAR_OUT):
    r = np.hypot(XG, YG)
    band = (r >= r_in) & (r <= r_out)
    front_mask = band & (YG > 0)
    rear_mask  = band & (YG < 0)
    rho_front = d[front_mask].mean()
    rho_rear  = d[rear_mask].mean()
    return (rho_rear - rho_front) / (rho_rear + rho_front)

A_field      = aniso_global(field)
A_v3         = aniso_global(v3)
A_v4         = aniso_global(v4)
A_v5         = aniso_global(v5)
A_near_field = aniso_near(field)
A_near_v3    = aniso_near(v3)
A_near_v4    = aniso_near(v4)
A_near_v5    = aniso_near(v5)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
edges = _edges
panels = [
    ('Field (Weinburd 2024)',          field, A_field, A_near_field),
    ('Hybrid v3 (uncalibrated)',       v3,    A_v3,    A_near_v3),
    ('Hybrid v4 (CMA-ES calibrated)',  v4,    A_v4,    A_near_v4),
    ('Hybrid v5 (CMA-ES calibrated)',  v5,    A_v5,    A_near_v5),
]

# Share a single (vmin, vmax) across all three panels so a viewer
# compensating for low color discrimination can compare panel
# positions on the colormap directly. vmax is the 99th percentile
# across the three normalized maps, which keeps a single hot pixel
# from compressing the dynamic range while still covering the v5
# preferred-spacing ring.
norms = [d / d.sum() for _, d, _, _ in panels]
vmax  = float(np.percentile(np.concatenate([n.ravel() for n in norms]), 99))
vmin  = 0.0

# Overlay circles for the near-zone band so the reader can see which
# pixels feed A_near.
_th = np.linspace(0, 2 * np.pi, 200)

ims = []
for ax, (title, _, A, A_near), dn in zip(axes, panels, norms):
    im = ax.imshow(dn.T, origin='lower',
                   extent=[edges[0], edges[-1], edges[0], edges[-1]],
                   cmap=SEQUENTIAL_CMAP, aspect='equal',
                   vmin=vmin, vmax=vmax)
    ims.append(im)
    ax.plot(0, 0, 'w^', markersize=11)
    ax.axhline(0, color='white', ls='--', alpha=0.5, lw=0.7)
    ax.axvline(0, color='white', ls='--', alpha=0.5, lw=0.7)
    for r in (R_NEAR_IN, R_NEAR_OUT):
        ax.plot(r * np.cos(_th), r * np.sin(_th),
                color='white', ls=':', lw=0.9, alpha=0.85)
    ax.set_xlabel('Left to right (cm)')
    ax.set_ylabel('Behind to ahead (cm)')
    ax.set_title(f'{title}\n'
                 f'$A_\\text{{global}} = {A:+.4f}$, '
                 f'$A_\\text{{near}} = {A_near:+.4f}$',
                 fontsize=10)

# One shared colorbar for all three panels to make the shared scale explicit.
cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.025, pad=0.02,
                    label='Relative density (shared scale)')

fig.suptitle('Body-centered neighbor density: field vs. uncalibrated vs. calibrated hybrid',
             fontsize=12, y=1.02)
# tight_layout would fight the shared colorbar geometry; the manual
# (fraction, pad) on fig.colorbar already produces a clean layout.

out = HERE / 'fig4_spatial_check.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
print(f'A_global  field={A_field:+.5f}  v3={A_v3:+.5f}  v4={A_v4:+.5f}  v5={A_v5:+.5f}')
print(f'A_near    field={A_near_field:+.5f}  v3={A_near_v3:+.5f}  '
      f'v4={A_near_v4:+.5f}  v5={A_near_v5:+.5f}'
      f'   (band [{R_NEAR_IN}, {R_NEAR_OUT}] cm)')
