"""

Pairwise permutation tests (FDR-corrected):

         group1        group2     mean1  ...    p_perm  p_fdr_bh  sig_fdr_0.05
0    bstem only  bstem to ctx  0.376995  ...  0.000999  0.000999          True
1    bstem only      ctx only  0.376995  ...  0.000999  0.000999          True
2  bstem to ctx      ctx only  0.284254  ...  0.000999  0.000999          True

Von-Economo classes p-values (spin):

'primary motor'            : 0.001998  ,
'association'              : 0.40559441,
'association'              : 0.81718282,
'primary/secondary sensory': 0.19280719,
'primary sensory'          : 0.73926074,
'limbic'                   : 0.56743257,
'insular'                  : 0.2967033

Regions of high hubness at the level of brainstem:
    mRT_merged_l    (*)
    SN_subregion2_l (*)
    mRT_merged_r    (*)
    RPa             (*)
    SN_subregion2_r (*)
    SN_subregion1_l (*)
    ION_r (*)
    SN_subregion1_r (*)
    ION_l           (*)
    VTA_PBP_r       (*)
    VTA_PBP_l       (*)
    RN_subregion1_l
    RN_subregion1_r
    PnO_PnC_r
    MiTg_spth_l
    MiTg_spth_r    (*)
    PnO_PnC_l
    ...

Regions of high hubness at the level of brainstem - after regression (nonlinear):
    mRT_merged_l    (*)
    SN_subregion2_l (*)
    mRT_merged_r    (*)
    RPa             (*)
    SN_subregion2_r (*)
    ION_r           (*)
    SN_subregion1_l (*)
    MnR
    ION_l           (*)
    SN_subregion1_r (*)
    RMg
    MiTg_spth_r     (*)
    VTA_PBP_r       (*)
    VTA_PBP_l       (*)
    MiTg_spth_l
    LC_l
    LC_r
    ...

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from scipy.io import loadmat
from functions import pval_cal
import matplotlib.pyplot as plt
from functions import plot_network
from neuromaps.images import load_data
from functions import vasa_null_Schaefer
from scipy.stats import spearmanr, pearsonr
from netneurotools.networks import consensus
from neuromaps.images import dlabel_to_gifti
from functions import show_on_surface_and_save
from functions import save_and_convert_to_cifti
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from netneurotools.datasets import fetch_schaefer2018
from nilearn.datasets import fetch_atlas_schaefer_2018
from palettable.colorbrewer.sequential import PuBuGn_9
from globals import path_sc, path_fc, path_results, path_medialwall
from globals import n_brainstem, n_cortex, path_atlas, path_dist_size

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

#------------------------------------------------------------------------------
# Distance
#------------------------------------------------------------------------------

to_drop = ['Sp_v', 'Sp_d', 'STh_subregion1_l',
           'STh_subregion2_l', 'HTH', 'STh_subregion2_r',
           'STh_subregion1_r', 'LG_l', 'LG_r', 'MG_l', 'MG_r',
           'L-Accumbens-area', 'R-Accumbens-area',
           'L-Amygdala', 'R-Amygdala', 'L-Hippocampus', 'R-Hippocampus',
           'L-Pallidum', 'R-Pallidum', 'L-Putamen', 'R-Putamen',
           'L-Caudate', 'R-Caudate', 'L-Thalamus-Proper', 'R-Thalamus-Proper',
           'L-Cereb-Ctx', 'R-Cereb-Ctx']
dist = pd.read_csv(path_dist_size + 'dist.csv', index_col = 0) # 'Unnamed: 0' becomes index
dist = dist.drop(index = to_drop, columns = to_drop)
dist = np.array(dist)
dist = dist.astype(float)

#------------------------------------------------------------------------------
# Structural connectome (SC)
#------------------------------------------------------------------------------

def scale(values, vmin = 0, vmax = 1, axis = None):
    '''
    Normalize log-transformed sc matrix between vmin and vmax.
    Larger value = more connected.
    '''
    min_val = values.min(axis = axis, keepdims = True)
    max_val = values.max(axis = axis, keepdims = True)
    s = (values - min_val) / (max_val - min_val)
    s = s * (vmax - vmin) + vmin
    return s

sc_subj = loadmat(path_sc + 'Finalconn_matrix_EndsOnly_Scheafer_19Sub_SGM_Cutoff007.mat')['CI_sift_scaledby10m'] # 19 - 483 - 483
sc_subj = sc_subj[:, idx_bc, :][:, :, idx_bc] # only include cortex-brainstem data: 19 - 458 - 458
sc_subj[np.isnan(sc_subj)] = 0
sc_subj[sc_subj != 0] = 1/-np.log(sc_subj[sc_subj != 0])
sc_subj[~np.isfinite(sc_subj)] = np.nan # replaces inf and -inf with NaN

for i in range(19):
    np.fill_diagonal(sc_subj[i, :, :], 0) # make diagonal 0 in each subject

sc_subj = scale(sc_subj, vmin = 0, vmax = 1, axis = (1, 2)) # normalization

np.save(path_results + 'sc_subj.npy', sc_subj) # save SC of individuals

sc_mean = np.mean(sc_subj, axis = 0) # average SC across participants

# Make consensus SC across participants
coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 458 by 3 - this is the coordination of parcels
distance = sklearn.metrics.pairwise_distances(coor)
hemiid = np.where(np.isin(info['hemisphere'], ["R", "M"]), 0, 1) # 0 = right or midline, 1 = left
sc_cns = consensus.struct_consensus(np.transpose(sc_subj, (1, 2, 0)),
                                   distance,
                                   hemiid[idx_bc].reshape(-1, 1))
consenses_vis = sc_cns.astype(float).copy()
consenses_vis[sc_cns == 1] = sc_mean[sc_cns == 1]
np.fill_diagonal(consenses_vis, 1) # make diagonal 1 for visualization only

#------------------------------------------------------------------------------
# Visualize the consensus sc matrix and save it as PNG
#------------------------------------------------------------------------------

plt.figure(figsize = (15, 15))
sns.heatmap(consenses_vis,
            vmin = 0, vmax = 1,
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.axhline(y = n_brainstem,
            color = 'black',
            linewidth = 0.8)
plt.axvline(x = n_brainstem,
            color = 'black',
            linewidth = 0.8)
plt.savefig(path_results + "heatmap_consensus_whole_brain.png", format = "png")
plt.show()

np.fill_diagonal(consenses_vis, np.nan) # make diagonal NaN

np.save(path_results + 'consensus.npy', consenses_vis) # save the consensus matrix

#------------------------------------------------------------------------------
# Define 3 network compartments
#------------------------------------------------------------------------------

sc_bsbs = consenses_vis[:n_brainstem, :n_brainstem]      # brainstem-brainstem
sc_bsctx = consenses_vis[:n_brainstem, n_brainstem:]     # brainstem-cortex
sc_ctxctx = consenses_vis[n_brainstem:, n_brainstem:]    # cortex-cortex

#------------------------------------------------------------------------------
# Visualize brainstem-cortex compartment of the consensus sc matrix and save it as PNG
#------------------------------------------------------------------------------

plt.figure(figsize = (1.899, 15))
sns.heatmap(sc_bsctx,
            vmin = 0, vmax = 1,
            cmap =  PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + "heatmap_consensus_ctxbs.png", format = "png")
plt.show()

#------------------------------------------------------------------------------
# Make the jitter bar plot to show differences in edge values across 3 network compartments
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize = (5, 7))
conn_blocks = [sc_bsbs,                 # brainstem-brainstem
               sc_bsctx,                # brainstem-cortex
               sc_ctxctx]               # cortex-cortex

labels      = ["bstem only",            # brainstem-brainstem
               "bstem to ctx",          # brainstem-cortex
               "ctx only"]              # cortex-cortex

color_map = {
    "bstem only":   "darkgreen",       # brainstem–brainstem
    "bstem to ctx": "cornflowerblue",  # brainstem–cortex
    "ctx only":     "#de7eaf"}         # cortex–cortex

vals_list = []
for f in conn_blocks:
    if f.shape[0] == f.shape[1]: # upper triangle
        vals = f[np.triu_indices(len(f), k = 1)][f[np.triu_indices(len(f), k = 1)] != 0]
    else:                        # rectangle - all edges included
        vals = f.flatten()
    vals = vals[vals != 0]       # drop zeros
    vals = vals[~np.isnan(vals)] # drop NaNs
    vals_list.append(vals)

x = np.arange(len(vals_list))
means = [np.nanmean(v) for v in vals_list]

for i, (lab, m) in enumerate(zip(labels, means)):
    ax.bar(x[i], # make the barplot
           m,
           width = 0.6,
           color = 'silver',
           edgecolor = 'silver',
           zorder = 1)

rng = np.random.default_rng(0)
for i, (name, vals) in enumerate(zip(labels,vals_list)): # put dots on top of bars
    x_jit = rng.normal(loc = x[i], scale = 0.04, size = len(vals))
    ax.scatter(x_jit, vals, s = 6, alpha = 0.25,
               color = color_map[name], zorder = 2)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 15)
ax.set_ylabel("edges of consensus sc")
ax.set_xlabel("network compartment")
sns.despine(top = True, bottom = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'jitter_bar_consensus_sc.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Statistics to assess significance of differences in edge weights for 3 compartments
#------------------------------------------------------------------------------

def perm_test_diff_means(x, y, n_perm = 1000, seed = 0):
    """
    Permutation test for difference in means: mean(x) - mean(y)
    Shuffles labels after pooling.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    obs = np.nanmean(x) - np.nanmean(y)

    pooled = np.concatenate([x, y])
    nx = len(x)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = np.mean(perm[:nx]) - np.mean(perm[nx:])

    p = pval_cal(obs, null, n_perm)
    return obs, p, null

# Pairwise comparisons
pairs = [
    ("bstem only", "bstem to ctx", vals_list[0], vals_list[1]),
    ("bstem only", "ctx only",     vals_list[0], vals_list[2]),
    ("bstem to ctx", "ctx only",   vals_list[1], vals_list[2]),
]

results = []
for (n1, n2, x, y) in pairs:
    obs, p, null = perm_test_diff_means(x, y, n_perm = 1000, seed = 0)
    results.append([n1, n2, np.mean(x), np.mean(y), obs, p])

res = pd.DataFrame(results, columns = ["group1", "group2",
                                       "mean1", "mean2",
                                       "mean_diff(1-2)",
                                       "p_perm"])
print(res)

# FDR correction across the 3 pairwise tests
rej, p_fdr, _, _ = multipletests(res["p_perm"].values,
                                 alpha = 0.05,
                                 method = "fdr_bh")
res["p_fdr_bh"] = p_fdr
res["sig_fdr_0.05"] = rej
print("\nPairwise permutation tests (FDR-corrected):\n")
print(res)

#------------------------------------------------------------------------------
# Cortical weighted-degree
#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[str(n_cortex) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

sc_bsctx_sth = np.nanmean(sc_bsctx, axis = 0) # calculate weighted degree

save_and_convert_to_cifti(atlas,
                          sc_bsctx_sth.reshape(n_cortex, 1),
                          n_cortex,
                          'cortex_weighted_degree',
                          path_results)

# Show on cortical maps and save as PNG
show_on_surface_and_save(sc_bsctx_sth.reshape(n_cortex, 1)+10e-100, n_cortex, 0,
                         np.max(sc_bsctx_sth.reshape(n_cortex, 1)),
                         path_results,'cortex_weighted_degree.png')

# Cortex-cortex connectivity
sc_ctxctx_sth = np.nanmean(sc_ctxctx, axis = 0) # calculate weighted degree

save_and_convert_to_cifti(atlas,
                          sc_ctxctx_sth.reshape(n_cortex, 1),
                          n_cortex,
                          'cortex_cortex_weighted_degree',
                          path_results)

# Show on cortical maps and save as PNG
show_on_surface_and_save(sc_ctxctx_sth.reshape(n_cortex, 1), n_cortex, 0,
                         np.max(sc_ctxctx_sth.reshape(n_cortex, 1)),
                         path_results,'cortex_cortex_weighted_degree.png')

# Show similarity of the two cortical maps
fig, ax = plt.subplots(figsize = (5, 5))
ax.scatter(sc_bsctx_sth, sc_ctxctx_sth, color = 'gray', s = 10, alpha = 0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_results + "similarity_of_sc_bsctx_and_sc_ctxctx_sth.svg", format = "svg")
plt.show()

r_sp_of_sc_bsctx_and_sc_ctxctx_sth = spearmanr(sc_bsctx_sth, sc_ctxctx_sth)[0]
r_pr_of_sc_bsctx_and_sc_ctxctx_sth = pearsonr(sc_bsctx_sth, sc_ctxctx_sth)[0]

schaefer = fetch_atlas_schaefer_2018(n_rois = n_cortex)
nspins = 1000
spins_1 = vasa_null_Schaefer(nspins) # create spin nulls
null_r_1 = np.zeros(nspins)

# Null values
for spin_ind in range(nspins):
    spinned_data_ctx = sc_bsctx_sth[spins_1[:, spin_ind]]
    null_r_1[spin_ind] = spearmanr(spinned_data_ctx, sc_ctxctx_sth)[0]

p_value_sim_ctxctx_bsctx = pval_cal(r_sp_of_sc_bsctx_and_sc_ctxctx_sth,
                                   null_r_1,
                                   nspins)

#------------------------------------------------------------------------------
# Brainstem is more connected to motor cortex - using spin test to assess significance
#------------------------------------------------------------------------------

def load_von_economo_atlas(path_in, nnodes):
    atlas_data = np.squeeze(scipy.io.loadmat(path_in + 'economo_Schaefer400.mat')['pdata'])
    return atlas_data - 1, ['primary motor',
                            'association',
                            'association',
                            'primary/secondary sensory',
                            'primary sensory',
                            'limbic',
                            'insular']

atlas_7Network_von, label_von_networks = load_von_economo_atlas(path_atlas, n_cortex)
num_labels = len(label_von_networks)

spins = vasa_null_Schaefer(nspins) # create spin nulls

# Actual/real values
network_specific_disease_measure = np.zeros((num_labels,))
for label_ind in range(num_labels):
    temp = []
    for roi_ind in range(n_cortex):
        if atlas_7Network_von[roi_ind] == label_ind:
            temp.append(sc_bsctx_sth[roi_ind,])
    network_specific_disease_measure[label_ind,] = np.nanmean(np.array(temp))

# Null values
network_specific_disease_measure_nulls = np.zeros((num_labels, nspins))
for spin_ind in range(nspins):
    spinned_data_ctx = sc_bsctx_sth[spins[:, spin_ind]]
    for label_ind in range(num_labels):
        temp = []
        for roi_ind in range(n_cortex):
            if atlas_7Network_von[roi_ind] == label_ind:
                temp.append(spinned_data_ctx[roi_ind])
        network_specific_disease_measure_nulls[label_ind, spin_ind] = np.nanmean(np.array(temp))

# Non-parametric p-value
p_value_spin = np.zeros((num_labels,))
for label_ind in range(num_labels):
    p_value_spin[label_ind] = pval_cal(network_specific_disease_measure[label_ind],
                                       network_specific_disease_measure_nulls[label_ind, :].flatten(),
                                       nspins)

# Visualization of the results as a jitter bar
bar_width = 0.75
y = np.arange(len(label_von_networks))
fig, ax = plt.subplots(figsize = (6, 3))
network_means = [np.nanmean(sc_bsctx_sth[atlas_7Network_von == label_ind])
                 for label_ind in range(len(label_von_networks))]
ax.barh(y, network_means, height = bar_width, color = 'silver', alpha = 0.7)

for label_ind in range(len(label_von_networks)):
    roi_values = sc_bsctx_sth[atlas_7Network_von == label_ind]
    jitter = (np.random.rand(len(roi_values)) - 0.5) * bar_width * 0.8
    ax.scatter(roi_values, y[label_ind] + jitter,
               color = 'gray', s = 10, alpha = 0.7)
ax.set_yticks(y)
ax.set_yticklabels(label_von_networks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_results + "von_economo.svg", format = "svg")
plt.show()

#------------------------------------------------------------------------------
# Brainstem weighted-degree
#------------------------------------------------------------------------------

a = np.ones((n_brainstem, n_brainstem)) # this is just a random input to plot edges
brainstem_weighted_degree  = np.nanmean(sc_bsctx, axis = 1) # calculate weighted degree

plot_network('brainstem_weighted_degree_coronal', a, coor[:n_brainstem, :], a,
             brainstem_weighted_degree,
             node_sizes = bs_voxels,
             views_orientation = 'horizontal',
             views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = 0.12657881504903845)

plot_network('brainstem_weighted_degree_saggital', a, coor[:n_brainstem, :], a,
             brainstem_weighted_degree,
             node_sizes = bs_voxels,
             views_orientation = 'horizontal',
             views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = 0.12657881504903845)

print(name_bstem[np.argsort(brainstem_weighted_degree)])

#------------------------------------------------------------------------------
# Uncomment if you want to visualize brainstem nuclei along their names
#------------------------------------------------------------------------------

# from functions import plot_brainstem_with_names_size
# plot_brainstem_with_names_size(0)

#------------------------------------------------------------------------------
# Regress out (nonlinear) volume from brainstem weighted-degree
#------------------------------------------------------------------------------

x = np.array(bs_voxels)
y = brainstem_weighted_degree

# Plot regressors as vectors - x
plt.figure(figsize = (1, 5))
sns.heatmap(x.reshape(-1, 1),
            vmin = 0, vmax = np.max(x),
            cmap =  PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + "vector_x_voxelsize_brainstem.png", format = "png")
plt.show()

# Plot regressors as vectors - y
plt.figure(figsize = (1, 5))
sns.heatmap(y.reshape(-1, 1),
            vmin = 0, vmax = np.max(y),
            cmap =  PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.savefig(path_results + "vector_y_voxelsize_brainstem_weighted_degree.png", format = "png")
plt.show()

plt.figure(figsize = (5, 5))
plt.scatter(x, y, color = 'silver')
print(spearmanr(x, y)) # r = 0.6331154137673345
plt.show()

x_t = np.log1p(x)
y_t = np.log1p(y)

reg = LinearRegression().fit(x_t.reshape(-1, 1), y_t)
brainstem_weighted_degree_resid = y_t - reg.predict(x_t.reshape(-1, 1))

plt.figure(figsize = (5, 5))
plt.scatter(x_t, brainstem_weighted_degree_resid, color = 'silver')
print(spearmanr(x_t, brainstem_weighted_degree_resid)) # r = -0.10767408654809159
plt.show()

plot_network('stha_size_voxels_reg_coronal',a, coor[:n_brainstem, :], a,
             brainstem_weighted_degree_resid,
             node_sizes = bs_voxels,
             views_orientation = 'horizontal',
             views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(brainstem_weighted_degree_resid))

plot_network('stha_size_voxels_reg_saggital', a, coor[:n_brainstem, :], a,
             brainstem_weighted_degree_resid,
             node_sizes = bs_voxels,
             views_orientation = 'horizontal',
             views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(brainstem_weighted_degree_resid))

print(name_bstem[np.argsort(brainstem_weighted_degree_resid)])

#------------------------------------------------------------------------------
# END