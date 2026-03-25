#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import plot_network
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from netneurotools.networks import consensus
from palettable.colorbrewer.sequential import PuBuGn_9
from globals import path_sc, path_fc, path_results, path_dist_size

#------------------------------------------------------------------------------
# Region information file
#------------------------------------------------------------------------------

info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv', index_col = 0) # 438 columns

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
dist_df = dist.drop(index=to_drop, columns=to_drop)
dist_df = dist_df.loc[names, names]
dist = dist_df.to_numpy(dtype = float)

#------------------------------------------------------------------------------
# Structural connectome (SC)
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

sc_subj = scale(sc_subj, vmin = 0, vmax = 1, axis = (1,2))

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

sc_bs_sub_d = a[bs0:sc0, sc0:cx0]

#------------------------------------------------------------------------------
# Save the dotted network
#------------------------------------------------------------------------------

sc_bs_sub_d_sth_ = np.mean(sc_bs_sub_d, axis = 0)
sc_bs_sub_d_temp = np.zeros((81, 81))
sc_bs_sub_d_temp[58:, :58] = sc_bs_sub_d.T

node_scores = np.ones(14+9+58)
plot_network('sc_bs_subcortex_d_networkcoronal', sc_bs_sub_d_temp,
             coor[:n_bs + n_sc + n_di,:], sc_bs_sub_d_temp, node_scores,
             node_sizes = node_scores*0.1,
             edge_cmap = PuBuGn_9.mpl_colormap, edge_vmin = 0,
             edge_vmax = np.max(sc_bs_sub_d_temp),
             views_orientation = 'horizontal', views = 'coronal',
             node_cmap = PuBuGn_9.mpl_colormap,linewidth = 2,
             node_vmin = 0,size_vmax = 0.3,
             node_vmax = np.max(1))

node_scores = np.ones(14+9+58)
plot_network('sc_bs_subcortex_d_network_saggital', sc_bs_sub_d_temp,
             coor[:n_bs + n_sc + n_di,:], sc_bs_sub_d_temp, node_scores,
             node_sizes = node_scores*0.1,
             edge_cmap = PuBuGn_9.mpl_colormap, edge_vmin = 0,
             edge_vmax = np.max(sc_bs_sub_d_temp),
             views_orientation = 'horizontal', views = 'saggital',
             node_cmap = PuBuGn_9.mpl_colormap,linewidth = 2,
             node_vmin = 0,size_vmax = 0.3,
             node_vmax = np.max(1))

#------------------------------------------------------------------------------
# Name the dotted network
#------------------------------------------------------------------------------

node_names = np.concatenate([
    name_bstem.to_numpy(),
    name_subcortex.to_numpy(),
    name_diencephalon.to_numpy()]).astype(str)

coords_3 = coor[:(n_bs + n_sc + n_di), :]
node_scores = np.ones(n_bs + n_sc + n_di, dtype = float)

def plot_nodes_with_labels_3d(coords, labels, values = None,
                              out_svg = None, elev = 0, azim = 0):
    """
    Simple 3D scatter with text labels for each node.
    coords : (N,3)
    labels : (N,)
    values : (N,) for coloring (optional)
    """
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection = "3d")

    if values is None:
        values = np.zeros(coords.shape[0], dtype = float)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("coolwarm")
    colors = cmap(norm(values))

    # Points
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s = 40, c = colors, alpha = 0.9, edgecolors = "k", linewidths = 0.3)

    # Labels
    for (x, y, z), lab in zip(coords, labels):
        if np.any(~np.isfinite([x, y, z])):
            continue
        ax.text(x, y, z, lab, fontsize = 7, color = "k")

    ax.view_init(elev = elev, azim = azim)
    ax.axis("off")

    # Colorbar
    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink = 0.6, ax = ax, pad = 0.1)
    cbar.set_label("node value", fontsize = 12)

    plt.tight_layout()
    if out_svg is not None:
        plt.savefig(out_svg, dpi = 300)
    plt.show()

plot_nodes_with_labels_3d(
    coords_3, node_names, values = node_scores,
    out_svg = os.path.join(path_results, "nodes_labels_coronal.svg"),
    elev = 0, azim = 0)

plot_nodes_with_labels_3d(
    coords_3, node_names, values = node_scores,
    out_svg = os.path.join(path_results, "nodes_labels_sagittal.svg"),
    elev = 0, azim = 90)

#------------------------------------------------------------------------------
# do some form of visialization - heatmap
#------------------------------------------------------------------------------

X = sc_bs_sub_d.copy() # shape: (n_bs, n_subcortex+n_diencephalon)

g = sns.clustermap(X,
                    row_cluster = True,
                    col_cluster = False,          # keep (subcortex | diencephalon) order
                    cmap = PuBuGn_9.mpl_colormap,
                    vmin = 0, vmax = np.max(X),   # for z-scored view
                    xticklabels = np.concatenate((name_subcortex, name_diencephalon)),
                    yticklabels = name_bstem,     # show labels; set False if too crowded
                    figsize = (12, 14))
ax = g.ax_heatmap
ax.axvline(n_subcortex, color = "black", linewidth = 0.8)
plt.savefig(path_results + "heatmap_rowcluster_bs_sub_d.png",
            dpi = 300, bbox_inches = "tight")
plt.show()

# If you need the new brainstem order:
row_order = g.dendrogram_row.reordered_ind
name_bstem_ordered = np.array(name_bstem)[row_order]
X_ordered = X[row_order, :]

#------------------------------------------------------------------------------
# Heatmap visualization - combine left and right nuclei for bilateral ones
#------------------------------------------------------------------------------

def base_name(x: str) -> str:
    """
    Remove hemisphere tags from names.
    Handles:
      - suffix: _l, _r, _L, _R
      - prefix: L-, R-
    """
    x = str(x)
    x = re.sub(r'^[LR]-', '', x)
    x = re.sub(r'_(l|r)$', '', x, flags=re.IGNORECASE)
    return x

def collapse_lr_rows_cols(X, row_names, col_names):
    """
    Collapse L/R for BOTH rows and columns by base_name(), using mean or sum.
    Returns:
      X_collapsed (np.ndarray),
      rows_collapsed (np.ndarray),
      cols_collapsed (np.ndarray)
    """
    df = pd.DataFrame(X, index=[base_name(n) for n in row_names],
                         columns=[base_name(n) for n in col_names])
    df = df.groupby(level=0).agg(np.nanmean)
    df = df.T.groupby(level=0).agg(np.nanmean).T
    return df.to_numpy(dtype=float), df.index.to_numpy(), df.columns.to_numpy()

X = sc_bs_sub_d.copy() # (n_bs, n_subcortex+n_diencephalon)
col_names = np.concatenate([name_subcortex.to_numpy().astype(str),
                            name_diencephalon.to_numpy().astype(str)])
row_names = name_bstem.to_numpy().astype(str)

# Collapse L/R on BOTH axes
X_bi, row_bi, col_bi = collapse_lr_rows_cols(X, row_names, col_names)

# We also need the NEW boundary index after collapsing subcortex cols
sub_bi = pd.Index([base_name(n) for n in name_subcortex.to_numpy().astype(str)]).unique().to_numpy()
die_bi = pd.Index([base_name(n) for n in name_diencephalon.to_numpy().astype(str)]).unique().to_numpy()

# Ensure columns are ordered as [subcortex_bilat, diencephalon_bilat]
col_order = list(sub_bi) + [c for c in die_bi if c not in set(sub_bi)]

# Reorder X columns to this block order
df_bi = pd.DataFrame(X_bi, index = row_bi, columns = col_bi)
df_bi = df_bi.loc[:, col_order]

X_bi = df_bi.to_numpy()
col_bi = df_bi.columns.to_numpy()
row_bi = df_bi.index.to_numpy()

new_boundary = len(sub_bi) # where diencephalon starts (after collapsing)
g = sns.clustermap(X_bi,
                    row_cluster = True,
                    col_cluster = False,
                    cmap = PuBuGn_9.mpl_colormap,
                    vmin = 0,
                    vmax = np.max(X),
                    xticklabels = col_bi,
                    yticklabels = row_bi,
                    figsize = (14, 14))
g.ax_heatmap.axvline(new_boundary, color = "black", linewidth = 0.8)
plt.savefig(path_results + "heatmap_rowcluster_bs_sub_d_BILAT.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END