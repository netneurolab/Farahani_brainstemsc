#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import plot_network
from netneurotools.networks import consensus
from functions import show_on_surface_and_save
from palettable.colorbrewer.sequential import PuBuGn_9
from globals import (path_sc, path_fc, path_results, 
                     path_dist_size, n_cortex, n_brainstem)

#------------------------------------------------------------------------------
# Region information file
#------------------------------------------------------------------------------

info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv', index_col=0) # 438 columns

# 1:58 brainstem # 59:67 diencephalic # 68:74 subcortical # 76:475 cortical # 476:483 subcortcal

idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_subcortex = info.query("structure == 'subcortex'").index.values
idx_diencephalon = info.query("structure == 'diencephalon'").index.values

idx_bc = np.concatenate((idx_bstem, idx_subcortex, idx_diencephalon, idx_ctx))

name_bstem = info.labels[idx_bstem]
name_subcortex = info.labels[idx_subcortex]
name_diencephalon = info.labels[idx_diencephalon]

names = info.labels[idx_bc].to_numpy()

bstem_voxels = info.nvoxels[idx_bstem]
subcortex_voxels = info.nvoxels[idx_subcortex]
diencephalon_voxels = info.nvoxels[idx_diencephalon]

n_subcortex = len(name_subcortex)
n_diencephalon = len(name_diencephalon)

coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 481 by 3 - this is the coordination of parcels

#------------------------------------------------------------------------------
# Distance
#------------------------------------------------------------------------------

to_drop = ['Sp_v', 'Sp_d', 
           'L-Cereb-Ctx', 'R-Cereb-Ctx']
dist = pd.read_csv(path_dist_size + 'dist.csv', index_col = 0)
dist_df = dist.drop(index=to_drop, columns = to_drop)
dist_df = dist_df.loc[names, names]
dist = dist_df.to_numpy(dtype=float)

#------------------------------------------------------------------------------
# SC
#------------------------------------------------------------------------------

def scale(values, vmin = 0, vmax = 1, axis = None):
    '''
    Normalize log-transformed sc matrix between vmin and vmax.
    Larger value = more connected.
    '''
    min_val = values.min(axis=axis, keepdims = True)
    max_val = values.max(axis=axis, keepdims = True)
    s = (values - min_val) / (max_val - min_val)
    s = s * (vmax - vmin) + vmin
    return s

sc_subj = loadmat(path_sc + 'Finalconn_matrix_EndsOnly_Scheafer_19Sub_SGM_Cutoff007.mat')['CI_sift_scaledby10m']
sc_subj = sc_subj[:,idx_bc,:][:,:, idx_bc]
sc_subj[np.isnan(sc_subj)] = 0
sc_subj[sc_subj != 0] = 1/-np.log(sc_subj[sc_subj != 0])
sc_subj[~np.isfinite(sc_subj)] = np.nan

# Normalization
for i in range(19):
    np.fill_diagonal(sc_subj[i,:,:], 0)

sc_subj = scale(sc_subj, vmin=0, vmax=1, axis = (1,2))

# Average across people
sc_mean = np.mean(sc_subj, axis = 0)

# Consensus across people
coor = info[['x', 'y', 'z']].values
coor = coor[idx_bc, :] # 458 by 3 - this is the coordination of parcels
distance = sklearn.metrics.pairwise_distances(coor)
hemiid = np.where(np.isin(info['hemisphere'], ["R", "M"]), 0, 1)
sc_cns = consensus.struct_consensus(np.transpose(sc_subj, (1, 2, 0)),
                                   distance,
                                   hemiid[idx_bc].reshape(-1, 1))
a = sc_cns.astype(float).copy()
a[sc_cns == 1] = sc_mean[sc_cns == 1]
np.fill_diagonal(a, 1)

#------------------------------------------------------------------------------
# visualize the consensus sc matrix
#------------------------------------------------------------------------------

plt.figure(figsize = (15, 15)) # consensus
sns.heatmap(a,
            vmin = 0, vmax = 1,
            cmap = PuBuGn_9.mpl_colormap,
            xticklabels = False,
            yticklabels = False,
            cbar = False)
plt.axhline(y=n_brainstem,
            color='black',
            linewidth=0.8)
plt.axvline(x=n_brainstem,
            color='black',
            linewidth=0.8)
plt.axhline(y=n_brainstem + n_subcortex,
            color='black',
            linewidth=0.8)
plt.axvline(x=n_brainstem + n_subcortex,
            color='black',
            linewidth=0.8)
plt.axhline(y=n_brainstem + n_subcortex + n_diencephalon,
            color='black',
            linewidth=0.8)
plt.axvline(x=n_brainstem + n_subcortex + n_diencephalon,
            color='black',
            linewidth=0.8)
plt.savefig(path_results + "heatmap_SC_with_subcortex.png", format = "png")
plt.show()

#------------------------------------------------------------------------------
# Compare values in different network compartments
#------------------------------------------------------------------------------

n_bs = len(idx_bstem)
n_sc = len(idx_subcortex)
n_di = len(idx_diencephalon)

bs0 = 0
sc0 = n_bs
di0 = n_bs + n_sc
cx0 = n_bs + n_sc + n_di

sc_bsbs   = a[bs0:sc0, bs0:sc0]
sc_subsub = a[sc0:di0, sc0:di0]
sc_dd     = a[di0:cx0, di0:cx0]
sc_ctxctx = a[cx0:, cx0:]

sc_bs_ctx  = a[bs0:sc0, cx0:]
sc_sub_ctx = a[sc0:di0, cx0:]
sc_d_ctx   = a[di0:cx0, cx0:]
sc_bs_sub  = a[bs0:sc0, sc0:di0]
sc_bs_d    = a[bs0:sc0, di0:cx0]
sc_sub_d   = a[sc0:di0, di0:cx0]

conn_blocks = [sc_bsbs,
               sc_subsub,
               sc_dd,
               sc_ctxctx,
               sc_bs_ctx,
               sc_sub_ctx,
               sc_d_ctx,
               sc_bs_sub,
               sc_bs_d,
               sc_sub_d]

labels      = ["bstem only", "subcortex only",
               "diencephalon only",
               "ctx only",
               "bstem-cortex",
               "subcortex-cortex",
               "diencephalon-cortex",
               "bstem-subcortex",
               "bstem-diencephalon",
               "subcortex-diencephalon"]

color_map = {
    "bstem only"            : "darkgreen",
    "subcortex only"        : "orange",
    "diencephalon only"     : "red",
    "ctx only"              : "#de7eaf",
    "bstem-cortex"          : "cornflowerblue",
    "subcortex-cortex"      : "pink",
    "diencephalon-cortex"   : "blue",
    "bstem-subcortex"       : "black",
    "bstem-diencephalon"    : "gray",
    "subcortex-diencephalon": "green"}

vals_list = []
for f in conn_blocks:
    if f.shape[0] == f.shape[1]:  # square: use upper triangle
        vals = f[np.triu_indices(len(f), k = 1)][f[np.triu_indices(len(f), k = 1)] != 0]
    else:                         # rectangular: use all elements
        vals = f.flatten()
    vals = vals[vals != 0]       # drop zeros
    vals = vals[~np.isnan(vals)] # drop NaNs
    vals_list.append(vals)

fig, ax = plt.subplots(figsize = (12, 7))

x = np.arange(len(vals_list))
means = [np.nanmean(v) for v in vals_list]
for i, (lab, m) in enumerate(zip(labels, means)):
    ax.bar(x[i], m, width = 0.6, color = 'silver', edgecolor = 'silver', zorder = 1)

rng = np.random.default_rng(0) # for reproducibility
for i, (name, vals) in enumerate(zip(labels, vals_list)):
    x_jit = rng.normal(loc = x[i], scale = 0.04, size = len(vals))
    ax.scatter(x_jit, vals, s = 6, alpha = 0.25, color = color_map[name], zorder = 2)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 15)
ax.set_ylabel("weighted sc")
ax.set_xlabel("connection type")
ax.set_title("Structural connectivity by connection type")
sns.despine(top = True, bottom = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'jitter_sc_with_subcortex.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Save the cortical hubness maps
#------------------------------------------------------------------------------

sc_bs_ctx_sth = np.mean(sc_bs_ctx, axis = 0)
sc_sub_ctx_sth = np.mean(sc_sub_ctx, axis = 0)
sc_d_ctx_sth = np.mean(sc_d_ctx, axis = 0)

# show on brain and save as png
show_on_surface_and_save(sc_bs_ctx_sth.reshape(n_cortex, 1), n_cortex, 0,
                         np.max(sc_bs_ctx_sth.reshape(n_cortex, 1)),
                         path_results,'sc_bs_ctx_sth.png')

show_on_surface_and_save(sc_sub_ctx_sth.reshape(n_cortex, 1), n_cortex, 0,
                         np.max(sc_sub_ctx_sth.reshape(n_cortex, 1)),
                         path_results,'sc_sub_ctx_sth.png')

show_on_surface_and_save(sc_d_ctx_sth.reshape(n_cortex, 1), n_cortex, 0,
                         np.max(sc_d_ctx_sth.reshape(n_cortex, 1)),
                         path_results,'sc_d_ctx_sth.png')

#------------------------------------------------------------------------------
# Save the brainstem hubness maps
#------------------------------------------------------------------------------

a = np.zeros((n_brainstem, n_brainstem))
sc_bs_ctx_sth_ = np.mean(sc_bs_ctx, axis = 1)
plot_network('sc_bs_ctx_sth_coronal', a, coor[:n_brainstem,:], a, sc_bs_ctx_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_ctx_sth_))
plot_network('sc_bs_ctx_sth_saggital', a, coor[:n_brainstem,:], a, sc_bs_ctx_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_ctx_sth_))

sc_bsbs_sth_ =  np.mean(sc_bsbs, axis = 1)
plot_network('sc_bsbs_sth_coronal', a, coor[:n_brainstem,:], a, sc_bsbs_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'coronal', 
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bsbs_sth_))
plot_network('sc_bsbs_sth_saggital', a, coor[:n_brainstem,:], a, sc_bsbs_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bsbs_sth_))

sc_bs_d_sth_ = np.mean(sc_bs_d, axis = 1)
plot_network('sc_bs_d_sth_coronal', a, coor[:n_brainstem,:], a, sc_bs_d_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_d_sth_))
plot_network('sc_bs_d_sth_saggital', a, coor[:n_brainstem,:], a, sc_bs_d_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_d_sth_))

sc_bs_sub_sth_ = np.mean(sc_bs_sub, axis = 1)
plot_network('sc_bs_sub_sth_coronal', a, coor[:n_brainstem,:], a, sc_bs_sub_sth_,
             node_sizes = bstem_voxels,
             views_orientation='horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_sub_sth_))
plot_network('sc_bs_sub_sth_saggital', a, coor[:n_brainstem,:], a, sc_bs_sub_sth_,
             node_sizes = bstem_voxels,
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,
             node_vmin = 0,
             node_vmax = np.max(sc_bs_sub_sth_))

#------------------------------------------------------------------------------
# END