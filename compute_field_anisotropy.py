"""
Re-extract the field neighbor density map from the Weinburd .mat files
and compute the anisotropy index. Caches the array so future calibrations
don't need to re-run this.

This is a one-time computation — once we have A_field cached, the 4-metric
CMA-ES loss can compute A_sim per evaluation and compare.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io

BASE     = Path('/mnt/work_1/gest9386/CU_Boulder/CSCI-5423/Final_Project')
DATA_DIR = BASE / 'data'
OUT_DIR  = BASE / 'week5_cma'
OUT_DIR.mkdir(exist_ok=True)

# Column indices from week1 notebook
IDX_X, IDX_Y, IDX_FLAG, IDX_SPEED, IDX_THETA = 0, 1, 2, 3, 4
IDX_STATE = 8


def struct2data(data_struct):
    N_locusts, N_frames = data_struct.shape
    data_final = np.full((N_locusts, 9, N_frames), np.nan, dtype=np.float64)
    for i in range(N_locusts):
        for j in range(N_frames):
            feat = data_struct[i, j]['features']
            if feat.size > 0:
                data_final[i, :, j] = feat.flatten()
    return data_final


def load_clips(mat_path, rec_key):
    mat = scipy.io.loadmat(mat_path, squeeze_me=False)
    rec = mat[rec_key]
    clips = []
    for r in range(rec.shape[1]):
        rec_data = rec[0, r]
        data_field = rec_data['data']
        for c in range(data_field.shape[0]):
            clip_struct = data_field[c, 1]
            data_3d = struct2data(clip_struct)
            clips.append({'data': data_3d, 'recording_idx': r})
    return clips


def compute_field_density(data, radius=10.0, nbins=50, state_filter=None):
    """Identical to the week1 notebook's compute_neighbor_density_map."""
    N_locusts, _, N_frames = data.shape
    edges = np.linspace(-radius, radius, nbins + 1)
    hist  = np.zeros((nbins, nbins))
    for t in range(N_frames):
        x      = data[:, IDX_X, t]
        y      = data[:, IDX_Y, t]
        theta  = data[:, IDX_THETA, t]
        flag   = data[:, IDX_FLAG, t]
        speed  = data[:, IDX_SPEED, t]
        state  = data[:, IDX_STATE, t]
        valid = (flag == 1) & ~np.isnan(x) & ~np.isnan(y)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            continue
        pos = np.column_stack((x[valid_idx], y[valid_idx]))
        for ii, focal in enumerate(valid_idx):
            if np.isnan(theta[focal]) or speed[focal] <= 0:
                continue
            if state_filter is not None and state[focal] != state_filter:
                continue
            rel   = pos - pos[ii]
            dists = np.hypot(rel[:, 0], rel[:, 1])
            in_range = (dists > 0.1) & (dists < radius)
            if not in_range.any():
                continue
            rel_in = rel[in_range]
            rot = np.pi/2 - theta[focal]
            cr, sr = np.cos(rot), np.sin(rot)
            rx = rel_in[:, 0] * cr - rel_in[:, 1] * sr
            ry = rel_in[:, 0] * sr + rel_in[:, 1] * cr
            h, _, _ = np.histogram2d(rx, ry, bins=edges)
            hist += h
    return hist, edges


def anisotropy_index(density):
    """A = (rho_rear - rho_front) / (rho_rear + rho_front).
    Matches the convention in spatial_check.py."""
    H = density.shape[1]
    cy = H // 2
    rho_rear  = density[:, :cy].mean()   # y < 0  (behind)
    rho_front = density[:, cy:].mean()   # y > 0  (ahead)
    denom = rho_rear + rho_front
    return float((rho_rear - rho_front) / denom) if denom > 0 else 0.0


print('Loading Weinburd field data...')
clips_1 = load_clips(DATA_DIR / 'data_recording.mat',  'recording')
clips_2 = load_clips(DATA_DIR / 'data_recording2.mat', 'recording2')
all_clips = ([c for c in clips_1 if c['recording_idx'] <= 1]
             + [c for c in clips_2 if c['recording_idx'] >= 2])
# Match week1: use first 6 clips (Band 1) for the density map
sample = all_clips[:6]
print(f'Computing density map across {len(sample)} clips (walking + hopping)...')

# State_filter=None pulls all moving locusts (the week1 'moving' panel
# combined walking+hopping; reproduce that by filtering speed > 0 only,
# which the function already does).
combined = None
for i, clip in enumerate(sample):
    h_walk, edges = compute_field_density(clip['data'], state_filter=1)
    h_hop,  _     = compute_field_density(clip['data'], state_filter=2)
    h = h_walk + h_hop
    print(f'  clip {i+1}/{len(sample)}: {int(h.sum())} neighbor counts')
    combined = h if combined is None else combined + h

A_field = anisotropy_index(combined)
print(f'\nField anisotropy index A_field = {A_field:+.4f}')
print(f'Total neighbor counts: {int(combined.sum())}')

np.save(OUT_DIR / 'field_density.npy', combined)
with open(OUT_DIR / 'field_anisotropy.json', 'w') as f:
    json.dump({
        'anisotropy_index_A_field': A_field,
        'n_clips_used': len(sample),
        'total_counts': int(combined.sum()),
        'radius_cm': 10.0,
        'nbins': 50,
        'note': 'A > 0 means more density behind than ahead = forward void',
    }, f, indent=2)
print(f'Saved: {OUT_DIR / "field_density.npy"}')
print(f'Saved: {OUT_DIR / "field_anisotropy.json"}')
