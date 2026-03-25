"""

SC & FC:

    cortex-cortex
    SignificanceResult(statistic=0.31846670066901794, pvalue=0.0)
    brainstem-brainstem
    SignificanceResult(statistic=0.42208804977055336, pvalue=1.4647964282910902e-38)
    brainstem-cortex
    SignificanceResult(statistic=0.11601032986606509, pvalue=1.4905591581864694e-08)

real_r: ctxctx - bsbs - ctxbs
    array([0.3184667 , 0.42208805, 0.11601033]) (filip nulls)

p_val:
    array([0.001, 0.001, 0.001])

Comparing correlation values:

    cortex-cortex vs brainstem-cortex: Z = 9.884, p = 0 (two-sided)
    cortex-cortex vs brainstem-brainstem: Z = -3.461, p = 0.000537 (two-sided)
    brainstem-cortex vs brainstem-brainstem: Z = -8.377, p = 0 (two-sided)

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
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

#------------------------------------------------------------------------------
# Group-consensus SC and individual SC matrices
#------------------------------------------------------------------------------

consensus = np.load(path_results + 'consensus.npy') # load the consensus matrix
np.fill_diagonal(consensus, 0)

sc_subj = np.load(path_results + 'sc_subj.npy') # load SC of individuals
n_subj = sc_subj.shape[0]

consensus_ind = sc_subj.astype(float).copy()
consensus_ind[:, consensus == 1] = sc_subj[:, consensus == 1]
consensus_ind[:, consensus == 0] = 0

#------------------------------------------------------------------------------
# Functional connectome (FC)
#------------------------------------------------------------------------------

fc_matlab = loadmat(path_fc + 'mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
fcsubj = fc_matlab['C_BSwithHO']
fcsubj = np.delete(fcsubj, 10, axis = 2) # remove bad subject in sc
fc = np.mean(fcsubj[idx_bc,:,:][:, idx_bc,:], axis = 2)
np.fill_diagonal(fc, np.nan)

#------------------------------------------------------------------------------
# Define 3 network compartments
#------------------------------------------------------------------------------

sc_bsbs = np.asarray(consensus[:n_brainstem, :n_brainstem], dtype = float)
sc_bsctx = np.asarray(consensus[:n_brainstem, n_brainstem:], dtype = float)
sc_ctxctx = np.asarray(consensus[n_brainstem:, n_brainstem:], dtype = float)

fc_bsbs = np.asarray(fc[:n_brainstem, :n_brainstem], dtype = float)
fc_bsctx = np.asarray(fc[:n_brainstem, n_brainstem:], dtype = float)
fc_ctxctx = np.asarray(fc[n_brainstem:, n_brainstem:], dtype = float)

#------------------------------------------------------------------------------
# Brainstem–brainstem: remove lower triangle + SC = 0
#------------------------------------------------------------------------------

mask = np.tri(*sc_bsbs.shape, k = 0, dtype = bool) # lower triangle + diag
sc_bsbs_u   = np.where(mask, np.nan, sc_bsbs)
fc_bsbs_u = np.where(mask, np.nan, fc_bsbs)

# Remove SC == 0 and NaNs
keep = np.isfinite(sc_bsbs_u) & (sc_bsbs_u != 0)

flat_sc_bsbs = sc_bsbs_u[keep]
flat_fc_bsbs = fc_bsbs_u[keep]

#------------------------------------------------------------------------------
# Cortex–cortex: remove lower triangle + SC = 0
#------------------------------------------------------------------------------

mask = np.tri(*sc_ctxctx.shape, k = 0, dtype = bool)
sc_ctxctx_u   = np.where(mask, np.nan, sc_ctxctx)
fc_ctxctx_u = np.where(mask, np.nan, fc_ctxctx)

# Remove SC == 0 and NaNs
keep = np.isfinite(sc_ctxctx_u) & (sc_ctxctx_u != 0)

flat_sc_ctxctx = sc_ctxctx_u[keep]
flat_fc_ctxctx = fc_ctxctx_u[keep]

#------------------------------------------------------------------------------
# Brainstem–cortex: no symmetry, just drop SC = 0
#------------------------------------------------------------------------------

keep = (sc_bsctx != 0)
flat_sc_bsctx = sc_bsctx[keep]
flat_fc_bsctx = fc_bsctx[keep]

#------------------------------------------------------------------------------
# Stack all data from the three compartments
#------------------------------------------------------------------------------

sc_all = np.concatenate([flat_sc_ctxctx, flat_sc_bsbs, flat_sc_bsctx])
fc_all = np.concatenate([flat_fc_ctxctx, flat_fc_bsbs, flat_fc_bsctx])

labels = (
    ['cortex-cortex']       * len(flat_sc_ctxctx) +
    ['brainstem-brainstem'] * len(flat_sc_bsbs) +
    ['brainstem-cortex']    * len(flat_sc_bsctx))

sc_all   = np.asarray(sc_all)
fc_all = np.asarray(fc_all)
labels   = np.asarray(labels, dtype = object)

color_map = {
    'cortex-cortex'       : '#de7eaf',
    'brainstem-brainstem' : 'darkgreen',
    'brainstem-cortex'    : 'cornflowerblue'}

#------------------------------------------------------------------------------
# Figure : SC And FC
#------------------------------------------------------------------------------

plt.figure(figsize = (6,6))
for conn_type in ['cortex-cortex', 'brainstem-brainstem', 'brainstem-cortex']:
    idx = labels == conn_type
    if conn_type == 'cortex-cortex':
        plt.scatter(
            fc_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.1,
            color = color_map[conn_type],
            label = conn_type)
    if conn_type == 'brainstem-brainstem':
        plt.scatter(
            fc_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 0.8,
            color = color_map[conn_type],
            label = conn_type)
    else:
        plt.scatter(
            fc_all[idx],
            sc_all[idx],
            s = 10,
            alpha = 1,
            color = color_map[conn_type],
            label = conn_type)
    print(conn_type)
    print(spearmanr(fc_all[idx], sc_all[idx]))

sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'sc_fc.png', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Statistics - 1: is the correlation significant?
#------------------------------------------------------------------------------

SC_null = np.load(path_results + 'consenses_degree_lenght_preserving_nulls.npy')
#sc_n = np.load(path_results + 'bs_ctx_modular_nulls.npz', allow_pickle = True)
#SC_null= sc_n['array_data']

n_perm = 1000

null_3 =  np.zeros((3, n_perm))
for n_perm in range(n_perm):
    null_sc_bsbs = np.asarray(SC_null[n_perm, :n_brainstem, :n_brainstem], dtype = float)
    null_sc_bsctx = np.asarray(SC_null[n_perm,:n_brainstem, n_brainstem:], dtype = float)
    null_sc_ctxctx = np.asarray(SC_null[n_perm, n_brainstem:, n_brainstem:], dtype = float)

    null_fc_bsbs = np.asarray(fc[:n_brainstem, :n_brainstem], dtype = float)
    null_fc_bsctx = np.asarray(fc[:n_brainstem, n_brainstem:], dtype = float)
    null_fc_ctxctx = np.asarray(fc[n_brainstem:, n_brainstem:], dtype = float)

    # Brainstem–brainstem: remove lower triangle + SC = 0
    null_mask = np.tri(*null_sc_bsbs.shape, k = 0, dtype = bool) # lower triangle + diag
    null_sc_bsbs_u = np.where(null_mask, np.nan, null_sc_bsbs)
    null_fc_bsbs_u = np.where(null_mask, np.nan, null_fc_bsbs)
    
    # Remove SC == 0 and NaNs
    null_keep = np.isfinite(null_sc_bsbs_u) & (null_sc_bsbs_u != 0)
    
    null_flat_sc_bsbs = null_sc_bsbs_u[null_keep]
    null_flat_fc_bsbs = null_fc_bsbs_u[null_keep]

    # Cortex–cortex: remove lower triangle + SC = 0
    null_mask = np.tri(*null_sc_ctxctx.shape, k = 0, dtype = bool)
    null_sc_ctxctx_u = np.where(null_mask, np.nan, null_sc_ctxctx)
    null_fc_ctxctx_u = np.where(null_mask, np.nan, null_fc_ctxctx)
    
    # Remove SC == 0 and NaNs
    null_keep = np.isfinite(null_sc_ctxctx_u) & (null_sc_ctxctx_u != 0)
    
    null_flat_sc_ctxctx = null_sc_ctxctx_u[null_keep]
    null_flat_fc_ctxctx = null_fc_ctxctx_u[null_keep]
    
    # Brainstem–cortex: no symmetry, just drop SC = 0
    null_keep = (null_sc_bsctx != 0)
    null_flat_sc_bsctx = null_sc_bsctx[null_keep]
    null_flat_fc_bsctx = null_fc_bsctx[null_keep]

    # Stack all and build labels
    null_sc_all = np.concatenate([null_flat_sc_ctxctx, null_flat_sc_bsbs, null_flat_sc_bsctx])
    null_fc_all = np.concatenate([null_flat_fc_ctxctx, null_flat_fc_bsbs, null_flat_fc_bsctx])
    
    null_labels = (
        ['null_cortex-cortex']       * len(null_flat_sc_ctxctx) +
        ['null_brainstem-brainstem'] * len(null_flat_sc_bsbs) +
        ['null_brainstem-cortex']    * len(null_flat_sc_bsctx))

    null_sc_all = np.asarray(null_sc_all)
    null_fc_all = np.asarray(null_fc_all)
    null_labels = np.asarray(null_labels, dtype = object)

    c = 0
    for null_conn_type in ['null_cortex-cortex', 'null_brainstem-brainstem', 'null_brainstem-cortex',]:
        idx = null_labels == null_conn_type
        null_3[c, n_perm] = (spearmanr(null_fc_all[idx], null_sc_all[idx]))[0]
        c = c +1
    print(n_perm)
    
# figure : SC And Distance
real_r = np.zeros(3)
p_val = np.zeros(3)
c = 0
plt.figure(figsize = (6, 6))
for conn_type in ['cortex-cortex', 'brainstem-brainstem', 'brainstem-cortex']:
    idx = labels == conn_type
    real_r[c] = (spearmanr(fc_all[idx], sc_all[idx]))[0]
    p_val[c] = pval_cal(real_r[c], null_3[c], n_perm)
    c = c + 1

#------------------------------------------------------------------------------
# Statistics 2 - is Spearman correlation value higher in any network compartment?
#------------------------------------------------------------------------------

conn_types = ['cortex-cortex', 'brainstem-brainstem', 'brainstem-cortex']
r = {}
n = {}

for conn_type in conn_types:
    idx = labels == conn_type
    r[conn_type] = spearmanr(fc_all[idx], sc_all[idx]).correlation
    n[conn_type] = np.sum(idx)  # number of edges used in that compartment

def fisher_compare(r1, n1, r2, n2):
    # fisher z
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    Z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(Z)))
    return Z, p

pairs = [
    ('cortex-cortex', 'brainstem-cortex'),
    ('cortex-cortex', 'brainstem-brainstem'),
    ('brainstem-cortex', 'brainstem-brainstem')]

print("Spearman r:", r)

for a, b in pairs:
    Z, p = fisher_compare(r[a], n[a], r[b], n[b])
    print(f"{a} vs {b}: Z = {Z:.3f}, p = {p:.3g} (two-sided)")

#------------------------------------------------------------------------------
# Individuals data - compare whole matrix
#------------------------------------------------------------------------------

fc_subj = fcsubj[idx_bc,:,:][:, idx_bc,:] #(458, 458, 19)

sc_bsbs_ind = np.asarray(consensus_ind[:,:n_brainstem, :n_brainstem], dtype = float)
sc_bsctx_ind = np.asarray(consensus_ind[:,:n_brainstem, n_brainstem:], dtype = float)
sc_ctxctx_ind = np.asarray(consensus_ind[:,n_brainstem:, n_brainstem:], dtype = float)

fc_bsbs_ind = np.asarray(fc_subj[:n_brainstem, :n_brainstem,:], dtype = float)
fc_bsctx_ind = np.asarray(fc_subj[:n_brainstem, n_brainstem:,:], dtype = float)
fc_ctxctx_ind = np.asarray(fc_subj[n_brainstem:, n_brainstem:,:], dtype = float)


def sc_fc_corr_blocks(sc_mat, fc_mat, mode = 'brainstem-brainstem'):
    """
    Compute SC–FC Spearman correlation for one subject and one block.
    
    sc_mat, fc_mat: 2D arrays
    mode: 'bsbs', 'ctxctx', or 'bsctx'
    """
    sc = sc_mat.copy()
    fc = fc_mat.copy()
    if mode in ['bsbs', 'ctxctx']:
        # symmetric: keep only upper triangle (no diag)
        n = sc.shape[0]
        mask_lower = np.tri(n, n, k = 0, dtype = bool)  # lower + diag
        sc[mask_lower] = np.nan
        fc[mask_lower] = np.nan
    # Drop SC== 0 and NaNs
    keep = np.isfinite(sc) & np.isfinite(fc) & (sc != 0)
    sc_flat = sc[keep]
    fc_flat = fc[keep]
    r = spearmanr(sc_flat, fc_flat)[0]
    return r


corr_bsbs   = np.zeros(n_subj)
corr_bsctx  = np.zeros(n_subj)
corr_ctxctx = np.zeros(n_subj)

for s in range(n_subj):
    # brainstem–brainstem
    sc_bsbs_s = sc_bsbs_ind[s, :, :]         # (58, 58)
    fc_bsbs_s = fc_bsbs_ind[:, :, s]         # (58, 58)
    corr_bsbs[s] = sc_fc_corr_blocks(sc_bsbs_s, fc_bsbs_s, mode = 'brainstem–brainstem')

    # cortex–cortex
    sc_ctxctx_s = sc_ctxctx_ind[s, :, :]     # (400, 400)
    fc_ctxctx_s = fc_ctxctx_ind[:, :, s]     # (400, 400)
    corr_ctxctx[s] = sc_fc_corr_blocks(sc_ctxctx_s, fc_ctxctx_s, mode = 'cortex–cortex')

    # brainstem–cortex (rectangular, no triangle)
    sc_bsctx_s = sc_bsctx_ind[s, :, :]       # (58, 400)
    fc_bsctx_s = fc_bsctx_ind[:, :, s]       # (58, 400)
    corr_bsctx[s] = sc_fc_corr_blocks(sc_bsctx_s, fc_bsctx_s, mode = 'brainstem–cortex')


conn_types = ['cortex-cortex', 'brainstem-brainstem', 'brainstem-cortex']
data_map = {
    'cortex-cortex'      : corr_ctxctx,
    'brainstem-brainstem': corr_bsbs,
    'brainstem-cortex'   : corr_bsctx}

titles = {
    'cortex-cortex'        : 'cortex–cortex',
    'brainstem-brainstem'  : 'brainstem–brainstem',
    'brainstem-cortex'     : 'brainstem–cortex'}

x = np.arange(len(conn_types)) # positions: 0, 1, 2

plt.figure(figsize = (6, 5))
for i, ct in enumerate(conn_types):
    vals = np.asarray(data_map[ct])
    mean_val = np.nanmean(vals)
    x_jitter = x[i] + 0.06 * np.random.randn(len(vals))
    plt.scatter(
        x_jitter,
        vals,
        s = 32,
        alpha = 0.85,
        color = color_map[ct],
        edgecolor = 'k',
        linewidth = 0.3)
plt.xticks(x, [titles[ct] for ct in conn_types], rotation = 18, ha = 'right')
plt.ylabel('Spearman ρ (SC–FC)')
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'sc_fc_subjectwise_hubness.svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END