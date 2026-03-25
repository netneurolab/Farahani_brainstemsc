

"""

consenses_degree_lenght_preserving_nulls --> Permutation p-value: 0.001998001998001998

Max |sum(total_dominance)-R2_full| =  2.7755575615628914e-17

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from functions import pval_cal
from netneurotools import stats
import matplotlib.pyplot as plt
from functions import plot_network
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, zscore
from netneurotools.metrics import bct as bct_nn
from sklearn.linear_model import LinearRegression
from palettable.colorbrewer.sequential import PuBuGn_9
from globals import path_fc, path_results, n_brainstem, n_cortex

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

comm_mat = bct_nn.communicability_wei(consensus)

np.fill_diagonal(consensus, np.nan)

comm_ctxbs = comm_mat[n_brainstem:, :][:, :n_brainstem].T
sc_ctxbs = consensus[n_brainstem:, :][:, :n_brainstem].T

# Visualize the consensus sc matrix
plt.figure(figsize = (5, 15))
sns.heatmap(sc_ctxbs.T,
            vmin = 0, vmax = 0.01,
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels=False,
            yticklabels=False,
            cbar = False)
plt.savefig(path_results + 'SC_communicability_ctxbs.png', dpi = 300)
plt.show()

data_Sc_hubs = pd.DataFrame({
    'names': name_bstem,
    'FC_hubness': np.mean(sc_ctxbs, axis = 1)})

#------------------------------------------------------------------------------
# Functional connectome (FC)
#------------------------------------------------------------------------------

fc_matlab = loadmat(path_fc + 'mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
fcsubj = fc_matlab['C_BSwithHO']
fcsubj = np.delete(fcsubj, 10, axis = 2) # remove bad subject in sc
fc_ctxbs = np.mean(fcsubj[idx_bstem, :, :][:, idx_ctx, :], axis = 2)

# Visualize the functional connectivithy matrix
plt.figure(figsize = (5, 15))
sns.heatmap(fc_ctxbs.T,
            vmin = 0, vmax = 0.5,
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'fc_ctxbs.png', dpi = 300)
plt.show()

data_Fc_hubs = pd.DataFrame({
    'names': name_bstem,
    'FC_hubness': np.mean(fc_ctxbs, axis = 1)})

#------------------------------------------------------------------------------
# Compare brainstem to cortex FC and SC hubs (weighted degree) - 58 nodes on scatterplot
#------------------------------------------------------------------------------

plt.figure(figsize = (6, 6))
plt.scatter(np.mean(fc_ctxbs, axis = 1),
            np.mean(sc_ctxbs, axis = 1), color = 'silver', s = 5)
print(spearmanr(np.mean(fc_ctxbs, axis = 1),
                np.mean(sc_ctxbs, axis = 1))) # 0.3099967285867226
for x, y, name in zip(np.mean(fc_ctxbs, axis = 1),
                      np.mean(sc_ctxbs, axis = 1), name_bstem):
    plt.text(x, y, name, fontsize = 7, ha = 'left', va = 'bottom')
plt.xlabel("FC with cortex")
plt.ylabel("SC (consensus) with cortex")
plt.title("Similarity of FC and SC hubness of brainstem nuclei \
          when considering connectivity to cortex")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Compare cortex to brainstem FC and SC hubs (weighted degree) - 400 nodes on scatterplot
#------------------------------------------------------------------------------

plt.figure(figsize = (6, 6))
plt.scatter(np.mean(fc_ctxbs, axis = 0),
            np.mean(sc_ctxbs, axis = 0), color = 'silver', s = 5)
print(spearmanr(np.mean(fc_ctxbs, axis = 0),
                np.mean(sc_ctxbs, axis = 0))) # 0.10719145092570745
plt.xlabel("FC with brainstem")
plt.ylabel("SC (consensus) with brainstem")
plt.title("Similarity of FC and SC hubness of cortical regions \
          when considering connectivity to brainstem")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Distance and volume
#------------------------------------------------------------------------------

dist = np.load(path_results + 'distance_whole_brain.npy')
vol = np.load(path_results + 'volume_whole_brain.npy')

dist_ctxbs = dist[n_brainstem:, :][:, :n_brainstem].T

plt.figure(figsize = (5, 15))
sns.heatmap(dist_ctxbs.T,
            cmap = PuBuGn_9.mpl_colormap,
            vmin = 0,
            vmax = np.max(dist_ctxbs),
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + 'distance_ctxbs.png', dpi = 300)
plt.show()

vol_ctxbs = vol[n_brainstem:n_brainstem+n_cortex, :][:, :n_brainstem].T
plt.figure(figsize = (5, 15))
sns.heatmap(vol_ctxbs.T,
            cmap = PuBuGn_9.mpl_colormap,
            vmin = 0,
            vmax = 1802,
            xticklabels=False,
            yticklabels=False,
            cbar = False)
plt.savefig(path_results + 'vol_ctxbs.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Some scatterplots to explore the data
#------------------------------------------------------------------------------

def spearman_scatter(x, y, title, name_save):
    plt.figure(figsize = (7, 7))
    plt.scatter(x.flatten(),
                y.flatten(), c = 'silver', s = 40)
    r, p = spearmanr(x.flatten(),
                     y.flatten())
    plt.title(title)
    ax = plt.gca()
    ax.text(0.05, 0.95,
            f"r = {r:.2f}, p = {p:.3g}",
            transform=ax.transAxes,
            ha='left', va='top')
    sns.despine(top = True, right = True)
    plt.tight_layout()
    plt.savefig(path_results + name_save + '.svg', dpi = 300)
    plt.show()

spearman_scatter(sc_ctxbs[sc_ctxbs != 0], fc_ctxbs[sc_ctxbs != 0],
                 'all SC edges and FC edges', 'sc_fc_ctxbs')
spearman_scatter(sc_ctxbs[sc_ctxbs != 0], vol_ctxbs[sc_ctxbs != 0],
                 'all SC edges and volume edges', 'sc_volume_ctxbs')
spearman_scatter(sc_ctxbs[sc_ctxbs != 0], dist_ctxbs[sc_ctxbs != 0],
                 'all SC edges and distance edges', 'sc_distance_ctxbs')
spearman_scatter(fc_ctxbs, vol_ctxbs,
                 'all FC edges and volume edges', 'fc_volume_ctxbs')
spearman_scatter(fc_ctxbs, dist_ctxbs,
                 'all FC edges and distance edges', 'fc_distance_ctxbs')
spearman_scatter(np.mean(sc_ctxbs, axis = 1), np.mean(vol_ctxbs, axis = 1),
                 'SC strenght brainstem node with cortex versus average volume of that nuclei',
                 'sc_volume_ctxbs_brainstem_avg')
spearman_scatter(np.mean(sc_ctxbs, axis = 1), np.mean(dist_ctxbs, axis = 1),
                 'SC strenght brainstem node with cortex versus average distance of that nuclei from cortex',
                 'sc_distance_ctxbs_brainstem_avg')
spearman_scatter(np.mean(fc_ctxbs, axis = 1), np.mean(vol_ctxbs, axis = 1),
                 'FC strenght brainstem node with cortex versus average volume of that nuclei',
                 'fc_volume_ctxbs_brainstem_avg')
spearman_scatter(np.mean(fc_ctxbs, axis = 1), np.mean(dist_ctxbs, axis = 1),
                 'FC strenght brainstem node with cortex versus average distance of that nuclei from cortex',
                 'fc_distance_ctxbs_brainstem_avg')
spearman_scatter(np.mean(sc_ctxbs, axis = 0), np.mean(vol_ctxbs, axis = 0),
                 'SC strenght cortical node with brainstem versus average volume of that cortical node',
                 'sc_volume_ctxbs_cortical_avg')
spearman_scatter(np.mean(sc_ctxbs, axis = 0), np.mean(dist_ctxbs, axis = 0),
                 'SC strenght cortical node with brainstem versus average distance of that parcel from brainstem',
                 'sc_distance_ctxbs_cortical_avg')
spearman_scatter(np.mean(fc_ctxbs, axis = 0), np.mean(vol_ctxbs, axis = 0),
                 'FC strenght cortical node with brainstem versus average volume of that parcel',
                 'fc_volume_ctxbs_cortical_avg')
spearman_scatter(np.mean(fc_ctxbs, axis = 0), np.mean(dist_ctxbs, axis = 0),
                 'FC strenght cortical node with brainstem versus average distance of that parcel from brainstem',
                 'fc_distance_ctxbs_cortical_avg')

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
    'other'     : 'lightgrey'}

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

# Mask for dopaminergic nuclei
dopaminergic_mask = np.array([nm in dat_nuc for nm in name_bstem])

#------------------------------------------------------------------------------
# SC-FC coupling for the brainstem - cortex compartment - individual nuclei
# R2 with and without including SC when predicting FC!
#------------------------------------------------------------------------------

R2_model1 = np.zeros(n_brainstem) # distance + volume
R2_model2 = np.zeros(n_brainstem) # distance + volume + SC

for i in range(n_brainstem):
    # extract 400 cortical values for nucleus i
    y = fc_ctxbs[i, :]
    x_dist = dist_ctxbs[i, :]
    x_vol  = vol_ctxbs[i, :]
    x_sc   = comm_ctxbs[i, :]

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

# R2 - model 2
a = np.ones((n_brainstem, n_brainstem)) 
plot_network('R2_model_2', a, coor[:n_brainstem,:], a, R2_model2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(R2_model2), node_vmax = np.max(R2_model2))
plot_network('R2_model_2', a, coor[:n_brainstem,:], a, R2_model2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(R2_model2), node_vmax = np.max(R2_model2))

# Quantify the improvements in R2 from adding SC
delta_R2 = R2_model2 - R2_model1
np.save(path_results + 'delta_R2_bsctx.npy', delta_R2)

# Delta R2
plot_network('delta_R2_coronal', a, coor[:n_brainstem,:], a, delta_R2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(delta_R2), node_vmax = np.max(delta_R2))
plot_network('delta_R2_saggital', a, coor[:n_brainstem,:], a, delta_R2,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.min(delta_R2), node_vmax = np.max(delta_R2))

#------------------------------------------------------------------------------
# Visualize the results as a figure
#------------------------------------------------------------------------------

plt.figure(figsize = (9, 7))
plt.scatter(R2_model1, R2_model2, c = colors_nuc, s = 180)

lo = min(R2_model1.min(), R2_model2.min())
hi = max(R2_model1.max(), R2_model2.max())

plt.plot([lo, hi], [lo, hi], 'k--', linewidth = 1)
for x, y, name in zip(R2_model1, R2_model2, name_bstem):
    plt.text(x, y, name, fontsize = 7, ha = 'left', va = 'bottom')
plt.xlabel("R²: Distance + Volume")
plt.ylabel("R²: Distance + Volume + SC")
plt.title("Does SC Improve FC Prediction Per Brainstem Nucleus?")
sns.despine(top = True, right = True)

# Legend
legend_elements = [
    Line2D([0], [0], marker = 'o', color = 'w',
           label='serotonin', markerfacecolor = color_map['serotonin'], markersize = 8),
    Line2D([0], [0], marker = 'o', color = 'w',
           label='noradrenergic', markerfacecolor = color_map['noradr'], markersize = 8),
    Line2D([0], [0], marker = 'o', color = 'w',
           label='dopaminergic', markerfacecolor = color_map['dopamine'], markersize = 8),
    Line2D([0], [0], marker = 'o', color = 'w',
           label='cholinergic', markerfacecolor = color_map['cholin'], markersize = 8),
    Line2D([0], [0], marker = 'o', color = 'w',
           label='other', markerfacecolor = color_map['other'], markersize = 8)]

plt.legend(handles = legend_elements, bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.tight_layout()
plt.savefig(path_results  + 'ctxbs_fc_sc_coupling_R2_colored_nuclei.svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Statistical analysis to show dopaminergic nuclei have higher values for Delta-R2?
#------------------------------------------------------------------------------

delta_dopa = delta_R2[dopaminergic_mask]   # dopaminergic nuclei
delta_other = delta_R2[~dopaminergic_mask] # all others
observed_diff = delta_dopa.mean() - delta_other.mean()

n_perm = 1000
diff_null = np.zeros(n_perm)
for i in range(n_perm):
    perm = np.random.permutation(delta_R2)
    perm_dopa = perm[:len(delta_dopa)]
    perm_other = perm[len(delta_dopa):]
    diff_null[i] = perm_dopa.mean() - perm_other.mean()
    
p_perm = pval_cal(observed_diff, diff_null, n_perm)
print("Permutation p-value:", p_perm)

#------------------------------------------------------------------------------
# Visualize the results as a figure
#------------------------------------------------------------------------------

plt.figure(figsize = (5, 5))
x_other = np.zeros_like(delta_other)
x_dopa  = np.ones_like(delta_dopa)
jitter_other = x_other + np.random.normal(0, 0.03, size = len(delta_other))
jitter_dopa  = x_dopa  + np.random.normal(0, 0.03, size = len(delta_dopa))

plt.scatter(jitter_other, delta_other,
            color = 'silver', edgecolor = 'k', s = 50, label = 'Other nuclei')
plt.scatter(jitter_dopa, delta_dopa,
            color = 'orange', edgecolor = 'k', s = 50, label = 'Dopaminergic nuclei')

plt.xticks([0, 1], ["Other nuclei", "Dopaminergic"])
plt.ylabel("ΔR² (R²_full − R²_base)")
plt.title("Improvement in FC Prediction From Adding SC")
plt.text(0.5, max(delta_R2)*0.95,
         f"Δmean = {observed_diff:.4f}\nPermutation p = {p_perm:.4g}",
         ha = 'center')
plt.tight_layout()
plt.savefig(path_results  + 'ctxbs_fc_sc_coupling_R2_dopaminergic.svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Statistics - degree- and lenght- preserving nulls to assess role of SC in FC prediction
#------------------------------------------------------------------------------

# load SC nulls (already in bc space)
sc_ctxbs_null_all = np.load(path_results + 'consenses_degree_lenght_preserving_nulls.npy')  # (n_null,458,458)
# sc_n = np.load(path_results + 'bs_ctx_modular_nulls.npz', allow_pickle = True)
# sc_ctxbs_null_all = sc_n['array_data']

nspins = 1000

null_delta_R2 = np.zeros((nspins, n_brainstem))

for s in range(nspins):
    sc_ctxbs_null = sc_ctxbs_null_all[s,:,:]
    sc_ctxbs_null = bct_nn.communicability_wei(sc_ctxbs_null)[:n_brainstem, n_brainstem:]
    for i in range(n_brainstem):
        y = fc_ctxbs[i, :]
        x_dist = dist_ctxbs[i, :]
        x_vol  = vol_ctxbs[i, :]
        x_sc_null = sc_ctxbs_null[i, :]

        # baseline model (model 1) is the same; we can reuse R2_model1[i]
        X2_null = np.vstack([x_dist, x_vol, x_sc_null]).T

        lr2_null = LinearRegression(fit_intercept = True).fit(X2_null, y)
        R2_full_null = r2_score(y, lr2_null.predict(X2_null))

        null_delta_R2[s, i] = R2_full_null - R2_model1[i]

p_val_scatter = np.zeros(n_brainstem)
for i in range(n_brainstem):
    p_val_scatter[i] = pval_cal(delta_R2[i], null_delta_R2[:, i], nspins)

'''
from statsmodels.stats.multitest import multipletests
reject, pval_fdr, _, _ = multipletests(p_val_scatter, method='hommel')
'''
#------------------------------------------------------------------------------
# Visualize the results as a figure - signify the nuclei with sig results
#------------------------------------------------------------------------------

sig = p_val_scatter < 0.05 # define significance at p = 0.05

plt.figure(figsize = (7, 7))

# non-significant nuclei
plt.scatter(R2_model1[~sig], R2_model2[~sig], c = 'silver', s = 65,
    edgecolor = 'k', alpha = 0.6, label = 'n.s.')

# significant nuclei
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
plt.savefig(path_results  + 'ctxbs_fc_sc_coupling_R2_colored_significance.svg', dpi = 300)
plt.show()

# Show on brainstem nuclei - binary (sig & ~sig)
a = np.zeros((n_brainstem, n_brainstem))
plot_network('sig_delta_R2_coronal', a, coor[:n_brainstem,:], a,sig,
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'coronal', node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0, node_vmax = 1)
plot_network('sig_delta_R2_saggital', a, coor[:n_brainstem,:], a, sig,
             node_sizes = bc_voxels, views_orientation = 'horizontal',
             views = 'saggital', node_cmap = PuBuGn_9.mpl_colormap,
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
plt.savefig(path_results + 'delta_over_R2full_scatter.svg', dpi = 300)
plt.show()

a = np.zeros((n_brainstem, n_brainstem))
plot_network('delta_over_R2full_coronal_relative', a,
             coor[:n_brainstem, :], a, delta_over_R2full,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.nanmin(delta_over_R2full),
             node_vmax = np.nanmax(delta_over_R2full))
plot_network('delta_over_R2full_saggital_relative', a,
             coor[:n_brainstem, :], a, delta_over_R2full,
             node_sizes = bc_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = np.nanmin(delta_over_R2full),
             node_vmax = np.nanmax(delta_over_R2full))

#------------------------------------------------------------------------------
# R² contributions for Distance / Volume / SC - Dominance analysis
#------------------------------------------------------------------------------

sig_sc = (p_val_scatter < 0.05) # shape (58,) - define significance

starred_names = np.array(name_bstem, dtype = str).copy()
starred_names[sig_sc] = np.char.add(starred_names[sig_sc], "*")

starred_names = pd.Series(name_bstem).astype(str)
starred_names.loc[sig_sc] = starred_names.loc[sig_sc] + "*"
starred_names = starred_names.to_numpy()
# Save significance table
df_sig = pd.DataFrame({
    "nucleus": name_bstem,
    "p_sc_deltaR2": p_val_scatter,
    "sig_sc_deltaR2": sig_sc,
    "R2_base_dist_vol": R2_model1,
    "R2_full_dist_vol_sc": R2_model2,
    "delta_R2": delta_R2}).sort_values("p_sc_deltaR2")
df_sig.to_csv(path_results + "SC_significance_deltaR2_per_nucleus.csv", index = False)

# Dominance analysis per nucleus using netneurotools (y(FC) ~ [Distance, Volume, SC])
phi_dist = np.full(n_brainstem, np.nan)
phi_vol = np.full(n_brainstem, np.nan)
phi_sc = np.full(n_brainstem, np.nan)
R2_full_dom = np.full(n_brainstem, np.nan)

for i in range(n_brainstem):
    y = fc_ctxbs[i, :]
    d = dist_ctxbs[i, :]
    v = vol_ctxbs[i, :]
    s = comm_ctxbs[i, :]
    X_m = np.column_stack([zscore(d), zscore(v), zscore(s)])
    mm, _ = stats.get_dominance_stats(X_m, y,
                                      use_adjusted_r_sq = 'False',
                                      verbose = False, n_jobs = 1)
    td = np.asarray(mm["total_dominance"]).ravel()
    phi_dist[i], phi_vol[i], phi_sc[i] = td[0], td[1], td[2]
    R2_full_dom[i] = mm["full_r_sq"]

contrib = np.column_stack([phi_sc, phi_dist, phi_vol])
contrib_df = pd.DataFrame(
    contrib,
    index = name_bstem,
    columns = ["SC", "Distance", "Volume"])

check = (phi_dist + phi_vol + phi_sc) - R2_full_dom
print("Max |sum(total_dominance)-R2_full| = ", np.nanmax(np.abs(check)))

out_df = contrib_df.copy()
out_df["R2_full_dom"] = R2_full_dom

# attach ΔR²-based stats so everything is together
out_df["R2_base_dist_vol"] = R2_model1
out_df["R2_full_dist_vol_sc"] = R2_model2
out_df["delta_R2"] = delta_R2
out_df["p_sc_deltaR2"] = p_val_scatter
out_df["sig_sc_deltaR2"] = sig_sc

#------------------------------------------------------------------------------
# Visualize the results as a figure - signify the nuclei with sig results (for SC)
#------------------------------------------------------------------------------

plt.figure(figsize = (6, 14))
sns.heatmap(contrib_df,
            cmap = PuBuGn_9.mpl_colormap,
            vmin = np.nanmin(contrib),
            vmax = np.nanmax(contrib),
            linewidths = 0.2,
            linecolor = "white",
            yticklabels = starred_names)
plt.title("nucleus R² contribution")
plt.tight_layout()
plt.savefig(path_results + "heatmap_dominance_distance_volume_sc_ctxbs_starred.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END