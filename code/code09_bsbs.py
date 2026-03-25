#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
from netneurotools import stats
from sklearn.metrics import r2_score
from netneurotools.networks import consensus
from functions import plot_network, pval_cal
from netneurotools.metrics import bct as bct_nn
from sklearn.linear_model import LinearRegression
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
names = info.labels[idx_bc]
bc_voxels = info.nvoxels[idx_bstem]
coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 458 by 3 - this is the coordination of parcels

#------------------------------------------------------------------------------
# Group-consensus SC and compute weighted communicability
#------------------------------------------------------------------------------

consensus = np.load(path_results + 'consensus.npy') # load the consensus matrix
np.fill_diagonal(consensus, 1)

comm_all = bct_nn.communicability_wei(consensus)
comm_bsbs = comm_all[:n_brainstem, :n_brainstem]

np.fill_diagonal(comm_bsbs, 0)
plt.figure(figsize = (15, 15))
sns.heatmap(comm_bsbs,
            vmin = 0, vmax = np.max(comm_bsbs),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'SC_communicability_bsbs.png', dpi = 300)
plt.show()

np.fill_diagonal(comm_bsbs, np.nan) # make diagonal elements NaN

#------------------------------------------------------------------------------
# Functional connectome (FC)
#------------------------------------------------------------------------------

fc_matlab = loadmat(path_fc + 'mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
fcsubj = fc_matlab['C_BSwithHO']
fcsubj = np.delete(fcsubj, 10, axis = 2) # remove bad subject in sc
FC_subjects = fcsubj[idx_bstem, :, :][:, idx_bstem, :]

plt.figure(figsize = (15, 15))
sns.heatmap(np.mean(FC_subjects, axis = 2),
            vmin =0,
            vmax = np.max(np.mean(FC_subjects, axis = 2)),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'fc_bsbs.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Distance and volume
#------------------------------------------------------------------------------

dist = np.load(path_results + 'distance_whole_brain.npy')
vol = np.load(path_results + 'volume_whole_brain.npy')

dist_bsbs = dist[:n_brainstem, :][:, :n_brainstem].T
vol_bsbs = vol[:n_brainstem, :][:, :n_brainstem].T

plt.figure(figsize = (15, 15))
sns.heatmap(vol_bsbs,
            vmin = 0,
            vmax = np.max(vol_bsbs),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'vol_bsbs.png', dpi = 300)
plt.show()

plt.figure(figsize = (15, 15))
sns.heatmap(dist_bsbs,
            vmin = 0,
            vmax = np.max(dist_bsbs),
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'distance_bsbs.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Define some grouping and colors to show different neuromodulatory nuclei
#------------------------------------------------------------------------------

# Obtained from Hansen et al.
serotnin_nuc = ['MnR', 'RMg', 'ROb', 'RPa', 'CLi_RLi', 'DR_2020', 'PMnR']
net_nuc      = ['LC_r', 'LC_l', 'SubC_r', 'SubC_l']
dat_nuc      = ['SN_subregion1_l', 'SN_subregion1_r',
                'SN_subregion2_l', 'SN_subregion2_r', 'VTA_PBP_r', 'VTA_PBP_l']
col_nuc      = ['PTg_l', 'PTg_r', 'LDTg_CGPn_l', 'LDTg_CGPn_r']

# Colors for each neuromodulatory group
color_map = {
    'serotonin' : 'tab:green',
    'noradr'    : 'tab:blue',
    'dopamine'  : 'tab:orange',
    'cholin'    : 'tab:pink',
    'other'     : 'lightgrey' }

colors_nuc = []
groups_nuc = [] # legend
for nm in name_bstem:
    if nm in serotnin_nuc:
        colors_nuc.append(color_map['serotonin'])
        groups_nuc.append('serotonin')
    elif nm in net_nuc:
        colors_nuc.append(color_map['noradr'])
        groups_nuc.append('noradr')
    elif nm in dat_nuc:
        colors_nuc.append(color_map['dopamine'])
        groups_nuc.append('dopamine')
    elif nm in col_nuc:
        colors_nuc.append(color_map['cholin'])
        groups_nuc.append('cholin')
    else:
        colors_nuc.append(color_map['other'])
        groups_nuc.append('other')

#------------------------------------------------------------------------------
# SC-FC coupling for the brainstem - brainstem compartment - individual nuclei
# R2 with and without including SC when predicting FC!
#------------------------------------------------------------------------------

R2_model1 = np.zeros(n_brainstem)
R2_model2 = np.zeros(n_brainstem)
fc_bsbs = np.mean(FC_subjects, axis = 2) # 58 x 58

for i in range(n_brainstem):
    mask = ~np.isnan(comm_bsbs[i, :])
    
    x_dist = dist_bsbs[i, mask]
    x_vol  = vol_bsbs[i, mask]
    x_sc   = comm_bsbs[i, mask]
    y = fc_bsbs[i, mask]

    # Model 1 predictors
    X1 = np.vstack([x_dist, x_vol]).T

    # Model 2 predictors
    X2 = np.vstack([x_dist, x_vol, x_sc]).T

    # Fit both models
    lr1 = LinearRegression(fit_intercept = True).fit(X1, y)
    lr2 = LinearRegression(fit_intercept = True).fit(X2, y)

    # Compute R²
    R2_model1[i] = r2_score(y, lr1.predict(X1))
    R2_model2[i] = r2_score(y, lr2.predict(X2))

np.save(path_results + 'R2_bsbs.npy', R2_model2)

# R2 - model 2
a = np.ones((n_brainstem, n_brainstem)) 
plot_network('R2_model_2_bsbs_coronal', a,
             coor[:n_brainstem,:], a, R2_model2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(R2_model2), node_vmax = np.max(R2_model2))
plot_network('R2_model_2_bsbs_saggital', a,
             coor[:n_brainstem,:], a, R2_model2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(R2_model2), node_vmax = np.max(R2_model2))

# Quantify the improvements in R2 from adding SC
delta_R2 = R2_model2 - R2_model1
np.save(path_results + 'delta_R2_bsbs.npy', delta_R2)

# Delta R2
plot_network('delta_R2_bsbs_coronal', a,
             coor[:n_brainstem,:], a, delta_R2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(delta_R2), node_vmax = np.max(delta_R2))
plot_network('delta_R2_bsbs_saggital', a,
             coor[:n_brainstem,:], a, delta_R2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(delta_R2), node_vmax = np.max(delta_R2))

#------------------------------------------------------------------------------
# statistics
#------------------------------------------------------------------------------

# Load SC nulls (already in bc space)
sc_null_all = np.load(path_results + 'consenses_degree_lenght_preserving_nulls.npy')  # (n_null,458,458)
# sc_n = np.load(path_results + 'bs_ctx_modular_nulls.npz', allow_pickle = True)
# sc_ctxbs_null_all = sc_n['array_data']
nspins = 1000

null_delta_R2 = np.zeros((nspins, n_brainstem))

for s in range(nspins):
    sc_bsbs_null = sc_null_all[s,:,:]
    comm_bsbs_null = bct_nn.communicability_wei(sc_bsbs_null)[:n_brainstem, :n_brainstem]
    for i in range(n_brainstem):
        mask = np.arange(n_brainstem) != i
        y = fc_bsbs[i, mask]
        x_dist = dist_bsbs[i, mask]
        x_vol  = vol_bsbs[i, mask]
        x_sc_null = comm_bsbs_null[i, mask]

        # baseline model (model 1) is the same; we can reuse R2_model1[i]
        X2_null = np.vstack([x_dist, x_vol, x_sc_null]).T

        lr2_null = LinearRegression(fit_intercept = True).fit(X2_null, y)
        R2_full_null = r2_score(y, lr2_null.predict(X2_null))

        null_delta_R2[s, i] = R2_full_null - R2_model1[i]

p_val_scatter = np.zeros(n_brainstem)
for i in range(n_brainstem):
    p_val_scatter[i] = pval_cal(delta_R2[i], null_delta_R2[:, i], nspins)

#------------------------------------------------------------------------------
# Visualize the results as a figure - signify the nuclei with sig results
#------------------------------------------------------------------------------

sig = p_val_scatter < 0.05 # (58,)

plt.figure(figsize = (7, 7))

# Non-significant nuclei
plt.scatter(R2_model1[~sig], R2_model2[~sig], c = 'silver', s = 65,
    edgecolor = 'k', alpha = 0.6, label = 'n.s.')

# Significant nuclei
plt.scatter(R2_model1[sig], R2_model2[sig], c = 'red', s = 65,
            edgecolor = 'k', linewidths = 1.1, label = 'p < 0.05')

lo = min(R2_model1.min(), R2_model2.min())
hi = max(R2_model1.max(), R2_model2.max())
plt.plot([lo, hi], [lo, hi], 'k--', linewidth = 1)
for x, y, name in zip(R2_model1, R2_model2, name_bstem):
    plt.text(x, y, name, fontsize = 7, ha = 'left', va = 'bottom')

plt.xlabel("R²: Distance + Volume")
plt.ylabel("R²: Distance + Volume + SC")
plt.title("SC contribution to FC prediction per brainstem nucleus")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results  + 'bsbs_fc_sc_coupling_R2_colored_significance.svg', dpi = 300)
plt.show()

# Show on brainstem nuclei - binary (sig & ~sig)
plot_network('sig_R2_bsbs_coronal', a, coor[:n_brainstem, :], a, sig,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0, node_vmax = 1)
plot_network('sig_R2_bsbs_sagiotal', a, coor[:n_brainstem, :], a, sig,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0, node_vmax = 1)

#------------------------------------------------------------------------------
# Instead of absolute R2 improvement look into relative improvement of R2
# (compared to model 1)
#------------------------------------------------------------------------------

delta_over_R2full = delta_R2 / R2_model2
delta_over_R2base = delta_R2 / R2_model1

plt.figure(figsize = (6, 6))
plt.scatter(R2_model2, delta_over_R2full, c = colors_nuc, s = 180)
for x, y, name in zip(R2_model2, delta_over_R2full, name_bstem):
    plt.text(x, y, name, fontsize = 7, ha = 'left', va = 'bottom')
plt.axhline(0, color = 'k', linewidth = 0.8)
plt.xlabel("R²_full (dist + vol + SC)")
plt.ylabel("ΔR² / R²_full")
plt.title("Normalized SC contribution per brainstem nucleus")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'bsbs_delta_over_R2full_scatter.svg', dpi = 300)
plt.show()

a = np.zeros((n_brainstem, n_brainstem))
plot_network('bsbs_delta_over_R2full_coronal_relative', a,
             coor[:n_brainstem, :], a,
             delta_over_R2full, node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.nanmin(delta_over_R2full),
             node_vmax = np.nanmax(delta_over_R2full))
plot_network('bsbs_delta_over_R2full_saggital_relative', a,
             coor[:n_brainstem, :], a,
             delta_over_R2full,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.nanmin(delta_over_R2full), 
             node_vmax = np.nanmax(delta_over_R2full))

#------------------------------------------------------------------------------
# R² contributions for Distance / Volume / SC - Dominance analysis
#------------------------------------------------------------------------------
'''
starred_names = np.array(name_bstem, dtype = str).copy()
starred_names[sig] = np.char.add(starred_names[sig], "*")
'''

starred_names = pd.Series(name_bstem).astype(str)
starred_names.loc[sig] = starred_names.loc[sig] + "*"
starred_names = starred_names.to_numpy()

dom_total = np.full((58, 3), np.nan) # columns: [Distance, Volume, SC]
for i in range(n_brainstem):
    mask = np.arange(n_brainstem) != i
    y = fc_bsbs[i, mask]
    x_dist = dist_bsbs[i, mask]
    x_vol  = vol_bsbs[i, mask]
    x_sc   = comm_bsbs[i, mask]
    Xv = np.column_stack([zscore(x_dist), zscore(x_vol), zscore(x_sc)])
    mm, _ = stats.get_dominance_stats(Xv, y,
                                      use_adjusted_r_sq = False,
                                      verbose = False, n_jobs = 1)
    dom_total[i, :] = mm["total_dominance"]

heat_mat = pd.DataFrame(dom_total, index = starred_names,
                        columns = ["Distance", "Volume", "SC"])
plt.figure(figsize = (6, 14))
sns.heatmap(heat_mat,
            cmap = PuBuGn_9.mpl_colormap,
            vmin = np.nanmin(dom_total),
            vmax = np.nanmax(dom_total),
            linewidths = 0.2,
            linecolor = "white")
plt.title("nucleus R² contribution")
plt.tight_layout()
plt.savefig(path_results + "heatmap_dominance_distance_volume_sc_bsbs_starred.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END