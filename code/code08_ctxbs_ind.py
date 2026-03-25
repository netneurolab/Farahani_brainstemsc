#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from functions import plot_network
from netneurotools.metrics import bct as bct_nn
from palettable.colorbrewer.sequential import PuBuGn_9
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
bc_voxels = info.nvoxels[idx_bstem]

# Calculate average SC over all subjects - consensus
coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 458 by 3 - this is the coordination of parcels

#------------------------------------------------------------------------------
# SC individuals
#------------------------------------------------------------------------------

sc_subj = np.load(path_results + 'sc_subj.npy') # load SC of individuals

n_sub = len(sc_subj) # N = 19

sc_subj_comm = np.zeros_like(sc_subj)
for s in  range(n_sub):
    np.fill_diagonal(sc_subj[s,:,:], 1)
    sc_subj_comm[s, :, :] = bct_nn.communicability_wei(sc_subj[s,:,:])

sc = sc_subj_comm[:, :n_brainstem, :][:, :, n_brainstem:] # cortex-brainstem

#------------------------------------------------------------------------------
# FC individuals
#------------------------------------------------------------------------------

fc_matlab = loadmat(path_fc + 'mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
fcsubj = fc_matlab['C_BSwithHO']
fcsubj = np.delete(fcsubj, 10, axis = 2) # remove bad subject in sc
FC_subjects =(fcsubj[idx_bstem, :, :][:, idx_ctx, :])

#------------------------------------------------------------------------------
# Distance and volume
#------------------------------------------------------------------------------

dist = np.load(path_results + 'distance_whole_brain.npy')
vol = np.load(path_results + 'volume_whole_brain.npy')

dist = dist[:n_brainstem, :][:, n_brainstem:]
vol = vol[:n_brainstem, :][:, n_brainstem:]

#------------------------------------------------------------------------------
# FC - SC  brainstem-cortex
#------------------------------------------------------------------------------

def compute_R2(y, X):
    """Return R² of regression y ~ X using least squares."""
    # remove rows with NaNs

    beta, *_ = np.linalg.lstsq(X, y, rcond = None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# sc --> 19 by 58 by 58
R2_sc_static   = np.zeros((n_brainstem, n_sub))
R2_dist_static = np.zeros((n_brainstem, n_sub))
R2_vol_static  = np.zeros((n_brainstem, n_sub))
R2_full = np.zeros((n_brainstem, n_sub))

for s in range(n_sub):
    co_sub_static = sc[s,:,:]
    co_sub_static[np.isnan(co_sub_static)] = 0

    for i in range(n_brainstem):
        co_i   = co_sub_static[i, :]   # shape (58,)
        dist_i = dist[i, :]     # shape (58,)
        vol_i  = vol[i, :]      # shape (58,)

        # Dependent variable: co-fluctuation of node i with all others at time t
        temp = FC_subjects[:, :, s]
        y = temp[i, :]  # shape (58,)
        # Design matrix: [SC, distance, volume, uniform]
        ones = np.ones_like(y)
        X_full = np.column_stack([co_i, dist_i, vol_i, ones])
        # Full model R²
        R2f = compute_R2(y, X_full)
        R2_full[i, :] = R2f
        # Reduced models (drop one predictor at a time)
        # partial R²_k = R²_full - R²_reduced_without_k
        R2_red_sc = compute_R2(y, X_full[:, [1, 2, 3]])   # drop SC
        R2_red_dist = compute_R2(y, X_full[:, [0, 2, 3]]) # drop dist
        R2_red_vol = compute_R2(y, X_full[:, [0, 1, 3]])  # drop vol

        R2_sc_static[i, s] = R2f - R2_red_sc
        R2_dist_static[i, s] = R2f - R2_red_dist
        R2_vol_static[i, s] = R2f - R2_red_vol
    print(s)

plt.figure(figsize = (15, 15))
sns.heatmap(R2_sc_static,
            vmin = np.min(R2_sc_static),
            vmax = np.max(R2_sc_static),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = name_bstem,
            cbar = False)
plt.savefig(path_results +'R2_sc_static_allsubjects.svg', dpi = 300)
plt.show()

plt.figure(figsize = (15, 15))
sns.heatmap(np.mean(R2_sc_static, axis = 1).reshape(-1, 1),
            vmin = np.min(np.mean(R2_sc_static, axis = 1)),
            vmax = np.max(np.mean(R2_sc_static, axis = 1)),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels=False,
            #yticklabels = name_bstem,
            cbar = False)
plt.savefig(path_results +'mean_R2_sc_static_allsubjects.svg', dpi = 300)
plt.show()

a = np.ones((n_brainstem, n_brainstem))
plot_network('R2_sc_static_allsubjects_coronal', a,
             coor[:n_brainstem, :], a, np.mean(R2_sc_static, axis = 1),
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'coronal', node_cmap = PuBuGn_9.mpl_colormap)
plot_network('R2_sc_static_allsubjects_saggital', a,
             coor[:n_brainstem, :], a, np.mean(R2_sc_static, axis = 1),
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'saggital', node_cmap = PuBuGn_9.mpl_colormap)

plt.figure()
plt.scatter(np.mean(R2_sc_static, axis = 1),
            bc_voxels,
            color = 'silver', s = 10)
plt.title('cv and size')
sns.despine(top = True, right = True)
plt.tight_layout()

print('which regions have highest R2?')
arg_cv = pd.DataFrame({
    'regions':name_bstem,
    'R2 diff':np.mean(R2_sc_static, axis = 1)})

n_node, n_subj = R2_sc_static.shape

# Rank within each subject (column-wise)
ranks = np.zeros_like(R2_sc_static, dtype = float)
for s in range(n_subj):
    ranks[:, s] = rankdata(R2_sc_static[:, s], method = 'average')

# Average rank across subjects
mean_rank = ranks.mean(axis = 1)
plot_network('val_', a, coor[:n_brainstem,:], a, mean_rank,
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'coronal', node_cmap = PuBuGn_9.mpl_colormap)
plot_network('val_', a, coor[:n_brainstem,:], a, mean_rank,
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'saggital', node_cmap = PuBuGn_9.mpl_colormap)

#------------------------------------------------------------------------------
# END