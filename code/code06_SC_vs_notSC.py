"""

BSBS unique edges: 1653
BSBS n conn: 862 n absent: 791
BSBS MWU p=2.11e-50 | mean diff=0.0509
BSBS null p=0.03596 | null draws used=1000


BSCTX edges: 23200
BSCTX n conn: 2369 n absent: 20831
BSCTX MWU p=1.54e-73 | mean diff=0.0324
BSCTX null p=0.03397 | null draws used=1000

CTXCTX unique edges: 79800
CTXCTX n conn: 22995 n absent: 56805
CTXCTX MWU p=0 | mean diff=0.0680
CTXCTX null p=0.000999 | null draws used=1000

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from functions import pval_cal
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
# Group-consensus SC and compute weighted communicability
#------------------------------------------------------------------------------

sc_cons = np.load(path_results + 'consensus.npy') # load the consensus matrix
np.fill_diagonal(sc_cons, 1)

# Load SC nulls (already in bc space)
sc_null_all = np.load(path_results + 'consenses_degree_lenght_preserving_nulls.npy') # (n_null,458,458)
#sc_n = np.load(path_results + 'bs_ctx_modular_nulls.npz', allow_pickle = True)
#sc_null_all = sc_n['array_data']

#------------------------------------------------------------------------------
# Functional connectome (FC)
#------------------------------------------------------------------------------

fc_matlab = loadmat(path_fc + 'mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
fcsubj = fc_matlab['C_BSwithHO']
fcsubj = np.delete(fcsubj, 10, axis = 2) # remove bad subject in sc
fc_bc = np.mean(fcsubj[idx_bc,:,:][:, idx_bc,:], axis = 2)
np.fill_diagonal(fc_bc, np.nan)

#------------------------------------------------------------------------------
# Define network compartments
#------------------------------------------------------------------------------

sc_bsbs   = np.asarray(sc_cons[:n_brainstem, :n_brainstem], dtype = float)   # (58,58)
sc_bsctx  = np.asarray(sc_cons[:n_brainstem, n_brainstem:], dtype = float)   # (58,400)
sc_ctxctx = np.asarray(sc_cons[n_brainstem:, n_brainstem:], dtype = float)   # (400,400)

fc_bsbs   = np.asarray(fc_bc[:n_brainstem, :n_brainstem], dtype = float)     # (58,58)
fc_bsctx  = np.asarray(fc_bc[:n_brainstem, n_brainstem:], dtype = float)     # (58,400)
fc_ctxctx = np.asarray(fc_bc[n_brainstem:, n_brainstem:], dtype = float)     # (400,400)

#------------------------------------------------------------------------------
# Brainstem-brainstem: FC is more dominent when SC is present (unique edges)
#------------------------------------------------------------------------------

def _vectorize_upper(A, k = 1):
    """Upper triangle vectorization (unique undirected edges)."""
    iu = np.triu_indices(A.shape[0], k = k)
    return A[iu], iu

sc_vec_bsbs, iu_bsbs = _vectorize_upper(sc_bsbs, k = 1)
fc_vec_bsbs, _       = _vectorize_upper(fc_bsbs, k = 1)

mask_conn = (sc_vec_bsbs != 0) & np.isfinite(fc_vec_bsbs)
mask_abs  = (sc_vec_bsbs == 0) & np.isfinite(fc_vec_bsbs)

fc_conn_bsbs = fc_vec_bsbs[mask_conn]
fc_abs_bsbs  = fc_vec_bsbs[mask_abs]

# Mannwhitneyu test
u_stat, p_mwu_bsbs = mannwhitneyu(fc_conn_bsbs, fc_abs_bsbs, alternative = "two-sided")
m_conn_bsbs = float(np.mean(fc_conn_bsbs))
m_abs_bsbs  = float(np.mean(fc_abs_bsbs))
obs_diff_bsbs = m_conn_bsbs - m_abs_bsbs

print("BSBS unique edges:", len(fc_vec_bsbs))
print("BSBS n conn:", len(fc_conn_bsbs), "n absent:", len(fc_abs_bsbs))
print(f"BSBS MWU p={p_mwu_bsbs:.3g} | mean diff={obs_diff_bsbs:.4f}")

# Figure: jitter plot
rng = np.random.default_rng(0)
x0 = np.zeros(len(fc_conn_bsbs)) + rng.normal(0, 0.06, size = len(fc_conn_bsbs))
x1 = np.ones(len(fc_abs_bsbs)) + rng.normal(0, 0.06, size = len(fc_abs_bsbs))

plt.figure(figsize = (5, 8))
plt.scatter(x0, fc_conn_bsbs, s = 10, alpha = 0.35,
            color = "dimgray", edgecolor = "none")
plt.scatter(x1, fc_abs_bsbs, s = 10, alpha = 0.35,
            color = "dimgray",  edgecolor = "none")
plt.hlines(m_conn_bsbs, -0.18, 0.18, linewidth = 3)
plt.hlines(m_abs_bsbs, 0.82, 1.18, linewidth = 3)
plt.xticks([0, 1], ["SC-connected", "SC-not-connected"])
plt.ylabel("FC (BS–BS)")
ax = plt.gca()
ax.text(0.02, 0.98,
        f"MWU p = {p_mwu_bsbs:.3g}\nmean diff = {obs_diff_bsbs:.4f}\n"
        f"mean(conn) = {m_conn_bsbs:.4f}\nmean(not) = {m_abs_bsbs:.4f}",
        transform = ax.transAxes, ha = "left", va = "top")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "bsbs_FC_by_SC_presence_jitter.svg", dpi = 300)
plt.show()

# Non-parametric test: BSBS null test (keep FC fixed)
sc_null_bsbs = sc_null_all[:, :n_brainstem, :n_brainstem]  # (n_null, 58, 58)
null_diff_bsbs = np.full(sc_null_bsbs.shape[0], np.nan)

n_perm = 1000
for s in range(n_perm):
    scn_vec = sc_null_bsbs[s][iu_bsbs]
    m_conn = np.isfinite(fc_vec_bsbs) & (scn_vec != 0)
    m_abs  = np.isfinite(fc_vec_bsbs) & (scn_vec == 0)
    null_diff_bsbs[s] = np.mean(fc_vec_bsbs[m_conn]) - np.mean(fc_vec_bsbs[m_abs])

null_diff_bsbs = null_diff_bsbs[np.isfinite(null_diff_bsbs)]
p_null_bsbs = pval_cal(obs_diff_bsbs, null_diff_bsbs, n_perm)
print(f"BSBS null p={p_null_bsbs:.4g} | null draws used={len(null_diff_bsbs)}")

# Figure: null distibution
plt.figure(figsize = (6.0, 4.6))
plt.hist(null_diff_bsbs, bins = 40,
         color = "lightgrey", edgecolor = "none")
plt.axvline(obs_diff_bsbs, linewidth = 2)
plt.xlabel("Mean FC(conn) − Mean FC(not)\n(SC-null defines connectedness; FC fixed)")
plt.ylabel("Count")
plt.title(f"BS–BS null test: obs={obs_diff_bsbs:.4f}, p={p_null_bsbs:.4g}")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "bsbs_FC_by_SC_presence_null_histogram.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Brainstem-cortex: FC is more dominent when SC is present (rectangular edges)
#------------------------------------------------------------------------------

sc_vec_bsctx = sc_bsctx.ravel()
fc_vec_bsctx = fc_bsctx.ravel()

mask_conn = (sc_vec_bsctx != 0)
mask_abs  = (sc_vec_bsctx == 0)

fc_conn_bsctx = fc_vec_bsctx[mask_conn]
fc_abs_bsctx  = fc_vec_bsctx[mask_abs]

# Mannwhitneyu test
u_stat, p_mwu_bsctx = mannwhitneyu(fc_conn_bsctx, fc_abs_bsctx, alternative = "two-sided")
m_conn_bsctx = float(np.mean(fc_conn_bsctx))
m_abs_bsctx  = float(np.mean(fc_abs_bsctx))
obs_diff_bsctx = m_conn_bsctx - m_abs_bsctx
print("BSCTX edges:", len(fc_vec_bsctx))
print("BSCTX n conn:", len(fc_conn_bsctx), "n absent:", len(fc_abs_bsctx))
print(f"BSCTX MWU p={p_mwu_bsctx:.3g} | mean diff={obs_diff_bsctx:.4f}")

# Figure: jitter plot
x0 = np.zeros(len(fc_conn_bsctx)) + rng.normal(0, 0.06, size = len(fc_conn_bsctx))
x1 = np.ones(len(fc_abs_bsctx)) + rng.normal(0, 0.06, size = len(fc_abs_bsctx))

plt.figure(figsize=(5,8))
plt.scatter(x0, fc_conn_bsctx, s = 8, alpha = 0.25,
            color = "dimgray", edgecolor = "none")
plt.scatter(x1, fc_abs_bsctx, s = 8, alpha = 0.25,
            color = "dimgray",  edgecolor = "none")
plt.hlines(m_conn_bsctx, -0.18, 0.18, linewidth = 3)
plt.hlines(m_abs_bsctx, 0.82, 1.18, linewidth = 3)
plt.xticks([0, 1], ["SC-connected", "SC-not-connected"])
plt.ylabel("FC (BS–CTX)")
ax = plt.gca()
ax.text(0.02, 0.98,
        f"MWU p = {p_mwu_bsctx:.3g}\nmean diff = {obs_diff_bsctx:.4f}\n"
        f"mean(conn) = {m_conn_bsctx:.4f}\nmean(not) = {m_abs_bsctx:.4f}",
        transform = ax.transAxes, ha = "left", va = "top")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "bsctx_FC_by_SC_presence_jitter.svg", dpi = 300)
plt.show()

# Non-parametric test: BSCTX null test (keep FC fixed)
sc_null_bsctx = sc_null_all[:, :58, 58:] # (n_null,58,400)
null_diff_bsctx = np.full(sc_null_bsctx.shape[0], np.nan)

for s in range(n_perm):
    scn_vec = sc_null_bsctx[s].ravel()
    m_conn = np.isfinite(fc_vec_bsctx) & (scn_vec != 0)
    m_abs  = np.isfinite(fc_vec_bsctx) & (scn_vec == 0)
    null_diff_bsctx[s] = np.mean(fc_vec_bsctx[m_conn]) - np.mean(fc_vec_bsctx[m_abs])

null_diff_bsctx = null_diff_bsctx[np.isfinite(null_diff_bsctx)]
p_null_bsctx = pval_cal(obs_diff_bsctx, null_diff_bsctx, n_perm)
print(f"BSCTX null p={p_null_bsctx:.4g} | null draws used={len(null_diff_bsctx)}")

# Figure: null distibution
plt.figure(figsize = (6.0, 4.6))
plt.hist(null_diff_bsctx, bins = 40,
         color = "lightgrey", edgecolor = "none")
plt.axvline(obs_diff_bsctx, linewidth = 2)
plt.xlabel("Mean FC(conn) − Mean FC(not)\n(SC-null defines connectedness; FC fixed)")
plt.ylabel("Count")
plt.title(f"BS–CTX null test: obs={obs_diff_bsctx:.4f}, p={p_null_bsctx:.4g}")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "bsctx_null_test_FCdiff.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# Cortex-cortex: FC is more dominent when SC is present (unique edges)
#------------------------------------------------------------------------------


sc_vec_ctxctx, iu_ctxctx = _vectorize_upper(sc_ctxctx, k = 1)
fc_vec_ctxctx, _         = _vectorize_upper(fc_ctxctx, k = 1)

mask_conn_ctxctx = (sc_vec_ctxctx != 0) & np.isfinite(fc_vec_ctxctx)
mask_abs_ctxctx  = (sc_vec_ctxctx == 0) & np.isfinite(fc_vec_ctxctx)

fc_conn_ctxctx = fc_vec_ctxctx[mask_conn_ctxctx]
fc_abs_ctxctx  = fc_vec_ctxctx[mask_abs_ctxctx]

# Mannwhitneyu test
u_stat_ctxctx, p_mwu_ctxctx = mannwhitneyu(fc_conn_ctxctx, fc_abs_ctxctx, alternative = "two-sided")
m_conn_ctxctx = float(np.mean(fc_conn_ctxctx))
m_abs_ctxctx  = float(np.mean(fc_abs_ctxctx))
obs_diff_ctxctx = m_conn_ctxctx - m_abs_ctxctx

print("CTXCTX unique edges:", len(fc_vec_ctxctx))
print("CTXCTX n conn:", len(fc_conn_ctxctx), "n absent:", len(fc_abs_ctxctx))
print(f"CTXCTX MWU p={p_mwu_ctxctx:.3g} | mean diff={obs_diff_ctxctx:.4f}")

# Figure: jitter plot
x0_ctxctx= np.zeros(len(fc_conn_ctxctx)) + rng.normal(0, 0.06, size = len(fc_conn_ctxctx))
x1_ctxctx = np.ones(len(fc_abs_ctxctx)) + rng.normal(0, 0.06, size = len(fc_abs_ctxctx))

plt.figure(figsize = (5, 8))
plt.scatter(x0_ctxctx, fc_conn_ctxctx, s = 10, alpha = 0.35,
            color = "#1b75bb", edgecolor = "none")
plt.scatter(x1_ctxctx, fc_abs_ctxctx, s = 10, alpha = 0.35,
            color = "#E3CBF2",  edgecolor = "none")
plt.hlines(m_conn_ctxctx, -0.18, 0.18, linewidth = 3, color = '#1b75bb')
plt.hlines(m_abs_ctxctx, 0.82, 1.18, linewidth = 3, color = '#9574c0')
plt.xticks([0, 1], ["SC-connected", "SC-not-connected"])
plt.ylabel("FC (CTX-CTX)")
ax = plt.gca()
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "ctxctx_FC_by_SC_presence_jitter.png", dpi = 300)
plt.show()

# Non-parametric test: CTXCTX null test (keep FC fixed)
sc_null_ctxctx = sc_null_all[:, n_brainstem:, n_brainstem:]  # (n_null, 400, 400)
null_diff_ctxctx = np.full(sc_null_ctxctx.shape[0], np.nan)

n_perm = 1000
for s in range(n_perm):
    scn_vec = sc_null_ctxctx[s][iu_ctxctx]
    m_conn = np.isfinite(fc_vec_ctxctx) & (scn_vec != 0)
    m_abs  = np.isfinite(fc_vec_ctxctx) & (scn_vec == 0)
    null_diff_ctxctx[s] = np.mean(fc_vec_ctxctx[m_conn]) - np.mean(fc_vec_ctxctx[m_abs])

null_diff_ctxctx = null_diff_ctxctx[np.isfinite(null_diff_ctxctx)]
p_null_ctxctx = pval_cal(obs_diff_ctxctx, null_diff_ctxctx, n_perm)
print(f"CTXCTX null p={p_null_ctxctx:.4g} | null draws used={len(null_diff_ctxctx)}")

# Figure: null distibution
plt.figure(figsize = (6.0, 4.6))
plt.hist(null_diff_ctxctx, bins = 40,
         color = "lightgrey", edgecolor = "none")
plt.axvline(obs_diff_ctxctx, linewidth = 2)
plt.xlabel("Mean FC(conn) − Mean FC(not)\n(SC-null defines connectedness; FC fixed)")
plt.ylabel("Count")
plt.title(f"CTX-CTX null test: obs={obs_diff_ctxctx:.4f}, p={p_null_ctxctx:.4g}")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + "ctxctx_FC_by_SC_presence_null_histogram.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END