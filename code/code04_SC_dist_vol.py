"""

SC - distance & sc - volume

Distance:
    ctx only
    SignificanceResult(statistic=-0.5500563614611923, pvalue=0.0)
    bstem to ctx
    SignificanceResult(statistic=-0.05263401901546485, pvalue=0.010399742783934404)
    bstem only
    SignificanceResult(statistic=-0.16129881570309298, pvalue=1.9355739428992206e-06)

Volume:
    ctx only
    SignificanceResult(statistic=0.026585700803804978, pvalue=5.5309760453338336e-05)
    bstem to ctx
    SignificanceResult(statistic=0.2545821110565461, pvalue=2.318521677382333e-36)
    bstem only
    SignificanceResult(statistic=0.3660919145228841, pvalue=9.895755930079864e-29)

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from globals import path_fc, path_results, n_brainstem

#------------------------------------------------------------------------------
# Region information file
#------------------------------------------------------------------------------

info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv', index_col = 0) # 438 columns
# 1:58 brainstem # 59:67 diencephalic # 68:74 subcortical # 76:475 cortical # 476:483 subcortcal
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))

name_bstem = info.labels[idx_bstem]
names = info.labels[idx_bc]
bs_voxels = info.nvoxels[idx_bstem]

coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 458 by 3 - this is the coordination of parcels

#-----------------------------------------------------------------------------
# Group-consensus SC
#------------------------------------------------------------------------------

sc_consensus = np.load(path_results + 'consensus.npy') # load the consensus matrix
np.fill_diagonal(sc_consensus, 0)

#------------------------------------------------------------------------------
# Distance and volume
#------------------------------------------------------------------------------

dist = np.load(path_results + 'distance_whole_brain.npy')
vol = np.load(path_results + 'volume_whole_brain.npy')

#------------------------------------------------------------------------------
# Define 3 network compartments
#------------------------------------------------------------------------------

sc_bsbs = np.asarray(sc_consensus[:n_brainstem, :n_brainstem], dtype = float)
sc_bsctx = np.asarray(sc_consensus[:n_brainstem, n_brainstem:], dtype = float)
sc_ctxctx = np.asarray(sc_consensus[n_brainstem:, n_brainstem:], dtype = float)

dist_bsbs = np.asarray(dist[:n_brainstem, :n_brainstem], dtype = float)
dist_bsctx = np.asarray(dist[:n_brainstem, n_brainstem:], dtype = float)
dist_ctxctx = np.asarray(dist[n_brainstem:, n_brainstem:], dtype = float)

vol_bsbs = np.asarray(vol[:n_brainstem, :n_brainstem], dtype = float)
vol_bsctx = np.asarray(vol[:n_brainstem, n_brainstem:], dtype = float)
vol_ctxctx = np.asarray(vol[n_brainstem:, n_brainstem:], dtype = float)

#------------------------------------------------------------------------------
# Brainstem–brainstem: remove lower triangle + SC = 0
#------------------------------------------------------------------------------

mask = np.tri(*sc_bsbs.shape, k = 0, dtype = bool) # lower triangle + diag
sc_bsbs_u   = np.where(mask, np.nan, sc_bsbs)
dist_bsbs_u = np.where(mask, np.nan, dist_bsbs)
vol_bsbs_u  = np.where(mask, np.nan, vol_bsbs)

# Remove SC == 0 and NaNs
keep = np.isfinite(sc_bsbs_u) & (sc_bsbs_u != 0)

flat_sc_bsbs   = sc_bsbs_u[keep]
flat_dist_bsbs = dist_bsbs_u[keep]
flat_vol_bsbs  = vol_bsbs_u[keep]

#------------------------------------------------------------------------------
# Cortex–cortex: remove lower triangle + SC = 0
#------------------------------------------------------------------------------

mask = np.tri(*sc_ctxctx.shape, k = 0, dtype = bool)
sc_ctxctx_u   = np.where(mask, np.nan, sc_ctxctx)
dist_ctxctx_u = np.where(mask, np.nan, dist_ctxctx)
vol_ctxctx_u  = np.where(mask, np.nan, vol_ctxctx)

# Remove SC == 0 and NaNs
keep = np.isfinite(sc_ctxctx_u) & (sc_ctxctx_u != 0)

flat_sc_ctxctx   = sc_ctxctx_u[keep]
flat_dist_ctxctx = dist_ctxctx_u[keep]
flat_vol_ctxctx  = vol_ctxctx_u[keep]

#------------------------------------------------------------------------------
# Brainstem–cortex: no symmetry, just drop SC = 0
#------------------------------------------------------------------------------

keep = (sc_bsctx != 0)

flat_sc_bsctx   = sc_bsctx[keep]
flat_dist_bsctx = dist_bsctx[keep]
flat_vol_bsctx  = vol_bsctx[keep]

#------------------------------------------------------------------------------
# Stack all data from the three compartments
#------------------------------------------------------------------------------

sc_all   = np.concatenate([flat_sc_ctxctx,   flat_sc_bsctx,   flat_sc_bsbs])
dist_all = np.concatenate([flat_dist_ctxctx, flat_dist_bsctx, flat_dist_bsbs])
vol_all  = np.concatenate([flat_vol_ctxctx,  flat_vol_bsctx,  flat_vol_bsbs])

labels = (
    ['ctx only']    * len(flat_sc_ctxctx) +
    ['bstem to ctx'] * len(flat_sc_bsctx) +
    ['bstem only'] * len(flat_sc_bsbs))

sc_all   = np.asarray(sc_all)
dist_all = np.asarray(dist_all)
vol_all  = np.asarray(vol_all)
labels   = np.asarray(labels, dtype = object)

# Drop NaNs / inf
valid = np.isfinite(sc_all) & np.isfinite(dist_all) & np.isfinite(vol_all)
sc_all   = sc_all[valid]
dist_all = dist_all[valid]
vol_all  = vol_all[valid]
labels   = labels[valid]

color_map = {
    "ctx only":     "#de7eaf",         # cortex–cortex
    "bstem to ctx": "cornflowerblue",  # brainstem–cortex
    "bstem only":   "darkgreen",       # brainstem–brainstem
}

#------------------------------------------------------------------------------
# Figure : SC And Distance
#------------------------------------------------------------------------------

plt.figure(figsize = (6, 6))
for conn_type in ['ctx only','bstem to ctx', 'bstem only']:
    idx = labels == conn_type
    if conn_type == 'ctx only':
        plt.scatter(
            dist_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.1,
            color = color_map[conn_type],
            label = conn_type)
    if conn_type == 'bstem only':
        plt.scatter(
            dist_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.8,
            color = color_map[conn_type],
            label = conn_type)
    else:
        plt.scatter(
            dist_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 1,
            color = color_map[conn_type],
            label = conn_type)
    print(conn_type)
    print(spearmanr(dist_all[idx], sc_all[idx],))

sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'sc_distance.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Figure : SC And Volume
#------------------------------------------------------------------------------

plt.figure(figsize = (6, 6))
for conn_type in ['ctx only','bstem to ctx', 'bstem only']:
    idx = labels == conn_type
    if conn_type == 'ctx only':
        plt.scatter(
            vol_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.1,
            color = color_map[conn_type],
            label = conn_type)
    if conn_type == 'bstem only':
        plt.scatter(
            vol_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.8,
            color = color_map[conn_type],
            label = conn_type)
    else:
        plt.scatter(
            vol_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 1,
            color = color_map[conn_type],
            label = conn_type)
    print(conn_type)
    print(spearmanr(vol_all[idx], sc_all[idx]))

sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'sc_volume.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END