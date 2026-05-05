"""
Regenerate Figure 3 (body-centered neighbor density maps) as a tight 4-panel
comparison -- Field | Vicsek | Pull | Anisotropic Vicsek. The original composite
had large amounts of whitespace and embedded sub-panels; this version renders
each panel from source data on a shared colorbar and a shared scale.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})

BASE = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
RMAX = 10.0
NBINS = 50
EDGES = np.linspace(-RMAX, RMAX, NBINS + 1)


def density_map(xy):
    """2D histogram of body-centered neighbor positions, normalized to density."""
    mask = (np.abs(xy[:, 0]) <= RMAX) & (np.abs(xy[:, 1]) <= RMAX)
    xy = xy[mask]
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[EDGES, EDGES])
    H = H / H.sum()
    return H


# --- Field --------------------------------------------------------------------
def field_neighbor_pairs():
    d = sio.loadmat(BASE / 'data' / 'data_recording.mat',
                    squeeze_me=True, struct_as_record=False)
    rec = d['recording']
    all_pairs = []
    for ri in range(len(rec)):
        data = rec[ri].data
        for clip in range(data.shape[0]):
            p = data[clip, 2]
            if hasattr(p, 'shape') and p.ndim == 2 and p.shape[1] == 2:
                all_pairs.append(p)
    return np.concatenate(all_pairs, axis=0)


# --- Sim helpers --------------------------------------------------------------
Lx = 99.44857450176137
Ly = 55.93982315724077
N  = 150
R_INT = 7.0
R_REP = 1.5
DT   = 0.04
NSTEPS = 2000
BURN = 500


def pairwise_delta(pos):
    delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    delta[:, :, 0] -= Lx * np.round(delta[:, :, 0] / Lx)
    delta[:, :, 1] -= Ly * np.round(delta[:, :, 1] / Ly)
    return delta


def collect_body_centered(pos, heading):
    """Return (x_local, y_local) for all in-range neighbor pairs at this frame."""
    delta = pairwise_delta(pos)
    d2 = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2
    mask = (d2 < RMAX ** 2) & (d2 > 0)
    xs, ys = [], []
    # rotate each focal agent's neighbor offsets into its body frame (heading up)
    cos_h = np.cos(-heading + np.pi / 2)  # heading-up => ahead = +y
    sin_h = np.sin(-heading + np.pi / 2)
    for i in range(N):
        idx = np.where(mask[i])[0]
        if len(idx) == 0:
            continue
        dx = delta[i, idx, 0]
        dy = delta[i, idx, 1]
        xs.append(cos_h[i] * dx - sin_h[i] * dy)
        ys.append(sin_h[i] * dx + cos_h[i] * dy)
    if not xs:
        return np.zeros((0, 2))
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])


def run_vicsek(eta=1.3, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [Lx, Ly], size=(N, 2))
    heading = rng.uniform(-np.pi, np.pi, size=N)
    speed = 11.76
    pairs = []
    for t in range(NSTEPS):
        delta = pairwise_delta(pos)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
        in_range = (dist < R_INT) & (dist > 0)
        unit = np.exp(1j * heading)
        weighted = (in_range * unit[np.newaxis, :]).sum(axis=1)
        avg_head = np.angle(weighted)
        noise = rng.uniform(-eta / 2, eta / 2, size=N)
        new_head = avg_head + noise
        rep_mask = (dist < R_REP) & (dist > 0)
        has_rep = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            new_head[has_rep] = (np.arctan2(rep_dy, rep_dx) + noise)[has_rep]
        heading = new_head
        pos[:, 0] = (pos[:, 0] + speed * np.cos(heading) * DT) % Lx
        pos[:, 1] = (pos[:, 1] + speed * np.sin(heading) * DT) % Ly
        if t > BURN and t % 2 == 0:
            pairs.append(collect_body_centered(pos, heading))
    return np.concatenate(pairs, axis=0)


def run_pull(eta=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [Lx, Ly], size=(N, 2))
    heading = rng.uniform(-np.pi, np.pi, size=N)
    speed = 11.76
    pairs = []
    for t in range(NSTEPS):
        delta = pairwise_delta(pos)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
        in_range = (dist < R_INT) & (dist > 0)
        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = (bearing_abs - heading[:, None] + np.pi) % (2 * np.pi) - np.pi
        w = np.cos(rel_bearing / 2) ** 2 * in_range
        unit = np.exp(1j * bearing_abs)
        ws = (w * unit).sum(axis=1)
        target = np.angle(ws)
        noise = rng.uniform(-eta / 2, eta / 2, size=N)
        heading = target + noise
        rep_mask = (dist < R_REP) & (dist > 0)
        has_rep = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            heading[has_rep] = np.arctan2(rep_dy, rep_dx)[has_rep] + noise[has_rep]
        pos[:, 0] = (pos[:, 0] + speed * np.cos(heading) * DT) % Lx
        pos[:, 1] = (pos[:, 1] + speed * np.sin(heading) * DT) % Ly
        if t > BURN and t % 2 == 0:
            pairs.append(collect_body_centered(pos, heading))
    return np.concatenate(pairs, axis=0)


def run_av(eta_base=1.1, lam=0.8, alpha=0.6, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [Lx, Ly], size=(N, 2))
    heading = rng.uniform(-np.pi, np.pi, size=N)
    v_min, v_max = 2.7, 11.76
    cone = np.pi / 3
    pairs = []
    for t in range(NSTEPS):
        delta = pairwise_delta(pos)
        dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
        in_range = (dist < R_INT) & (dist > 0)
        bearing_abs = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        rel_bearing = (bearing_abs - heading[:, None] + np.pi) % (2 * np.pi) - np.pi
        in_cone = np.abs(rel_bearing) < cone
        weights = np.where(in_cone, 1.0 + alpha, 1.0 - alpha) * in_range
        weights = np.clip(weights, 0, None)
        unit = np.exp(1j * heading)
        ws = (weights * unit[np.newaxis, :]).sum(axis=1)
        avg_head = np.angle(ws)
        f_forward = (in_range & in_cone).sum(axis=1) / np.maximum(in_range.sum(axis=1), 1)
        eta_eff = eta_base * (1.0 - lam * f_forward)
        noise = np.array([rng.uniform(-e / 2, e / 2) for e in eta_eff])
        new_head = avg_head + noise
        rep_mask = (dist < R_REP) & (dist > 0)
        has_rep = rep_mask.any(axis=1)
        if has_rep.any():
            rep_dx = -(rep_mask * delta[:, :, 0]).sum(axis=1)
            rep_dy = -(rep_mask * delta[:, :, 1]).sum(axis=1)
            new_head[has_rep] = (np.arctan2(rep_dy, rep_dx) + noise)[has_rep]
        heading = new_head
        order = np.abs(np.exp(1j * heading).mean())
        speed = v_min + (v_max - v_min) * order
        pos[:, 0] = (pos[:, 0] + speed * np.cos(heading) * DT) % Lx
        pos[:, 1] = (pos[:, 1] + speed * np.sin(heading) * DT) % Ly
        if t > BURN and t % 2 == 0:
            pairs.append(collect_body_centered(pos, heading))
    return np.concatenate(pairs, axis=0)


# --- Run ----------------------------------------------------------------------
print("Loading field data ...")
pairs_field = field_neighbor_pairs()
print("  field pairs:", pairs_field.shape)

print("Running Vicsek ..."); sys.stdout.flush()
pairs_vic = run_vicsek()

print("Running Pull ..."); sys.stdout.flush()
pairs_pull = run_pull()

print("Running Anisotropic Vicsek ..."); sys.stdout.flush()
pairs_av = run_av()

H_field = density_map(pairs_field)
H_vic   = density_map(pairs_vic)
H_pull  = density_map(pairs_pull)
H_av    = density_map(pairs_av)

# Field maps come from body-centered coords with ahead = +y already (convention
# in Weinburd et al. processing). Sim maps were rotated heading-up likewise.

# --- Plot ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
titles = [
    "Field (Weinburd 2024)",
    r"Vicsek  ($\eta=1.3$)",
    r"Pull  ($\eta=0.2$)",
    "Anisotropic Vicsek\n" r"($\eta_{\mathrm{base}}{=}1.1$, $\lambda{=}0.8$, $\alpha{=}0.6$)",
]
maps = [H_field, H_vic, H_pull, H_av]

# Each panel gets its own vmax (99th percentile of that panel's positive density)
# so within-panel anisotropy is visible on all four. A single shared colorbar is
# no longer meaningful; instead we label each colorbar per panel.
ims = []
for ax, H, t in zip(axes, maps, titles):
    vmax = np.quantile(H[H > 0], 0.99)
    im = ax.imshow(
        H.T,
        origin='lower',
        extent=(-RMAX, RMAX, -RMAX, RMAX),
        cmap='hot',
        vmin=0, vmax=vmax,
        aspect='equal',
    )
    ims.append(im)
    ax.axhline(0, color='white', lw=0.5, alpha=0.4, ls='--')
    ax.axvline(0, color='white', lw=0.5, alpha=0.4, ls='--')
    ax.scatter([0], [0], marker='^', s=60, color='white', edgecolor='black',
               linewidth=0.8, zorder=5)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("Left $\\leftarrow\\rightarrow$ Right (cm)", fontsize=9)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Behind $\\leftarrow\\rightarrow$ Ahead (cm)", fontsize=9)
for ax in axes[1:]:
    ax.set_yticklabels([])

# per-panel colorbar: one compact bar attached to each subplot
for ax, im in zip(axes, ims):
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.formatter.set_powerlimits((-3, 4))
    cb.update_ticks()

fig.savefig("figure3.png", dpi=200, bbox_inches="tight")
fig.savefig("figure3.pdf", bbox_inches="tight")
print("Saved figure3.png / figure3.pdf")
