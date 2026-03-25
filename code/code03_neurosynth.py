"""

Neurosynth analysis

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import matplotlib
import numpy as np
import pandas as pd
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from functions import vasa_null_Schaefer
from scipy.spatial.distance import pdist
from matplotlib.patches import Rectangle
from functions import show_on_surface_and_save
from netneurotools.metrics import bct as bct_nn
from palettable.colorbrewer.sequential import PuBuGn_9
from scipy.cluster.hierarchy import optimal_leaf_ordering, linkage, leaves_list
from globals import path_results, n_cortex, n_brainstem, path_fc, path_neurosynth

#------------------------------------------------------------------------------
# Region information file
#------------------------------------------------------------------------------

info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv', index_col=0) # 438 columns
# 1:58 brainstem # 59:67 diencephalic # 68:74 subcortical # 76:475 cortical # 476:483 subcortcal
idx_bstem = info.query("structure == 'brainstem'").index.values
name_bstem = info.labels[idx_bstem]

#------------------------------------------------------------------------------
# Group-consensus SC and compute weighted communicability
#------------------------------------------------------------------------------

sc_cns = np.load(path_results + 'consensus.npy') # load the consensus matrix
np.fill_diagonal(sc_cns, 1)
comm_all = bct_nn.communicability_wei(sc_cns)

#------------------------------------------------------------------------------
# Define the network compartment of interest
#------------------------------------------------------------------------------

sc_bsctx = comm_all[:n_brainstem, n_brainstem:]     # brainstem-cortex
#sc_bsctx = sc_cns[:n_brainstem, n_brainstem:]      # brainstem-cortex

#------------------------------------------------------------------------------
# Combine left and right hemispheres
#------------------------------------------------------------------------------

def combine_lr_profiles(profiles, names):
    """
    Combine rows that end with _L and _R into a single bilateral row.

    profiles : array (n_regions, n_features)  e.g., (58, 400)
    names    : list/array length n_regions

    Returns
    -------
    names_bilat : np.array of base names (no _L/_R)
    prof_bilat  : np.array (n_bilat, n_features)
    """
    df = pd.DataFrame(profiles.copy())
    df['name'] = np.array(names)

    # Strip trailing _L or _R
    df['base'] = df['name'].str.replace(r'_[lr]$', '', regex = True)

    feat_cols = df.columns[:-2]
    df_b = df.groupby('base')[feat_cols].agg('mean')
    names_bilat = df_b.index.to_numpy()
    prof_bilat  = df_b.to_numpy(dtype = float)

    return names_bilat, prof_bilat

def combine_lr_vector(values, names):
    """
    Combine a 1D vector (n_regions,) by averaging L/R pairs.
    """
    tmp = pd.DataFrame({'name': np.array(names), 
                        'val': np.array(values, dtype = float)})
    tmp['base'] = tmp['name'].str.replace(r'_[lr]$', '', regex = True)
    out = tmp.groupby('base')['val'].agg('mean')
    return out.index.to_numpy(), out.to_numpy()

name_bstem_bilat, sc_bsctx_bilat = combine_lr_profiles(sc_bsctx, name_bstem)

'''
# Show on cprtical maps and save as PNG
for i in range(33):
    show_on_surface_and_save(sc_bsctx_bilat[i,:].reshape(n_cortex, 1), n_cortex, 0 ,
                             np.max(sc_bsctx_bilat[i,:].reshape(n_cortex, 1)),
                             path_results, name_bstem_bilat[i] + '.png')
'''
#------------------------------------------------------------------------------
# Neurosynth
#------------------------------------------------------------------------------

neurosynth         = pd.read_csv(path_neurosynth + "neurosynth.csv")
neurosynth_data    = np.array(neurosynth)[:, 1:]
neurosynth_data    = neurosynth_data.astype(float)
neurosynth_columns = neurosynth.columns[1:]
# Create spins
nspins = 1000
spins = vasa_null_Schaefer(nspins)

alpha = 0.05 # significance level for p-value

# Calculate emprical and null correlation values
all_term_results = []
for term_idx in range(neurosynth_data.shape[1]): # 123 cognitive terms

    term_name = str(neurosynth_columns[term_idx])
    neurosynth_map = neurosynth_data[:, term_idx].astype(float)

    # Actual/real values
    r_emp = np.zeros(33)
    for i in range(33):
        r_emp[i] = spearmanr(sc_bsctx_bilat[i, :], neurosynth_map)[0]
            
    # Null values
    null_r = np.zeros((33, nspins))
    for s in range(nspins):
        neurosynth_map_spun = neurosynth_map[spins[:, s]]
        for i in range(33):
            null_r[i, s] = spearmanr(sc_bsctx_bilat[i, :], neurosynth_map_spun)[0]

    # p-value calculation - non-parametric
    p_emp = np.zeros(33)
    for i in range(33):
        p_emp[i] = pval_cal(r_emp[i], null_r[i, :], nspins)

    df = pd.DataFrame({
        "Nucleus": name_bstem_bilat,
        "r": r_emp,
        "p_spin": p_emp,
        "term_idx": term_idx,
        "term": neurosynth_columns[term_idx],
    })

    all_term_results.append(df)

    # Show progress
    print(term_idx)

all_results_df = pd.concat(all_term_results, ignore_index = True)

#------------------------------------------------------------------------------
# Only keep terms for which r > 0 and result is significant at least for one term
#------------------------------------------------------------------------------

df = all_results_df.copy()
df["sig_pos"] = (df["p_spin"] < alpha) & (df["r"] > 0)

from statsmodels.stats.multitest import multipletests
df["p_spin"] = pd.to_numeric(df["p_spin"], errors="coerce")
p_corrected = multipletests(df["p_spin"], method='fdr_bh')[1]
df["p_spin_corr"] = p_corrected


# Filter the results to keep: r > 0 and p-value < 0.05
terms_keep = df.groupby("term")["sig_pos"].any()
terms_keep = terms_keep[terms_keep].index.to_numpy()
df = df[df["term"].isin(terms_keep)].copy()

# R: r-values, S: sig+pos mask
R = df.pivot_table(index = "Nucleus", columns = "term",
                   values = "r", aggfunc = "mean")
S = df.pivot_table(index = "Nucleus", columns = "term",
                   values = "sig_pos", aggfunc = "max").astype(bool)

# Align
S = S.reindex(index = R.index, columns = R.columns)

# Cluster on R
A = R.to_numpy(dtype = float)
A = np.nan_to_num(A, nan = 0.0)

D_rows = pdist(A, metric = "correlation")
Zr = linkage(D_rows, method = "average")
Zr = optimal_leaf_ordering(Zr, D_rows)
ro = leaves_list(Zr)

D_cols = pdist(A.T, metric = "correlation")
Zc = linkage(D_cols, method = "average")
Zc = optimal_leaf_ordering(Zc, D_cols)
co = leaves_list(Zc)

R_ord = A[ro][:, co]
S_ord = S.to_numpy()[ro][:, co]

nuc_ord  = R.index.to_numpy()[ro]
term_ord = R.columns.to_numpy()[co]

#------------------------------------------------------------------------------
# Plot ordered heatmap and save the figure
#------------------------------------------------------------------------------

R_plot = R_ord.copy()
n_rows, n_cols = R_plot.shape
x = np.arange(n_cols + 1)
y = np.arange(n_rows + 1)

fig, ax = plt.subplots(figsize = (max(12, 0.35 * n_cols), max(8, 0.25 * n_rows)))
mesh = ax.pcolormesh(
    x, y, R_plot,
    cmap = PuBuGn_9.mpl_colormap,
    vmin = 0,
    vmax = np.max(R_plot),
    shading = "flat")
ax.invert_yaxis()
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels(nuc_ord, fontsize = 9)

ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_xticklabels(term_ord, fontsize = 8,
                   rotation = 90, ha = "center", va = "top")
ax.set_xlabel("Neurosynth term")
ax.set_ylabel("Brainstem nucleus")
ys, xs = np.where(S_ord)
for r, c in zip(ys, xs):
    ax.text(c + 0.5, r + 0.5, "*", ha = "center",
            va = "center", fontsize = 10, color = "black")
cbar = plt.colorbar(mesh, ax = ax, fraction = 0.046, pad = 0.03)
cbar.set_label("Spearman r")
for c in range(n_cols):
    col_rect = Rectangle((c, 0), 1, n_rows, fill = False, linewidth = 0)
    col_rect.set_label(f"col__{c}__{term_ord[c]}")
    ax.add_patch(col_rect)
plt.tight_layout()
matplotlib.rcParams["svg.fonttype"] = "none"
plt.savefig(path_results + "neurosynth_results_con_orig.svg", format = "svg")
plt.show()

#------------------------------------------------------------------------------
# END