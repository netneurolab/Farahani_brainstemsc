"""

SignificanceResult(statistic=0.2801378079916331, pvalue=0.033182227089786735)
SignificanceResult(statistic=0.3855039938668959, pvalue=0.0028040561720598244)
SignificanceResult(statistic=0.1596652883384558, pvalue=0.23122749446417504)

Spearman(bsctx, bsbs): r = 0.2801378079916331 perm p = 0.023976023976023976
Spearman(bsctx, voxels): r = 0.3855039938668959 perm p = 0.004995004995004995
Spearman(bsbs, voxels): r = 0.1596652883384558 perm p = 0.22077922077922077

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from globals import path_fc, path_results

#------------------------------------------------------------------------------
# Region information file
#------------------------------------------------------------------------------

info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv', index_col=0) # 438 columns
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
# Load data - delta R2 values
#------------------------------------------------------------------------------

bsctx = np.load(path_results + 'delta_R2_bsctx.npy')
bsbs = np.load(path_results + 'delta_R2_bsbs.npy')

#------------------------------------------------------------------------------
# Compare similarity of the two loaded vectors
#------------------------------------------------------------------------------

plt.figure(figsize = (7, 7))
plt.scatter(bsctx, bsbs, c = 'darkgreen', s = 65, edgecolor = 'k', linewidths = 1.1)
plt.xlabel("bsctx")
plt.ylabel("bsbs")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'fc_sc_ctxbs_bsbs_compare_delta_R2.svg',
            dpi = 300)
print(spearmanr(bsctx, bsbs)) # 0.2801378079916331
plt.show()

plt.figure(figsize = (7, 7))
plt.scatter(bsctx, bc_voxels, c = 'darkgreen', s = 65, edgecolor = 'k', linewidths = 1.1)
plt.xlabel("bsctx")
plt.ylabel("bc_voxels")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'fc_sc_ctxbs_bc_voxels_delta_R2.svg',
            dpi = 300)
print(spearmanr(bsctx, bc_voxels)) # 0.3855039938668959
plt.show()

plt.figure(figsize = (7, 7))
plt.scatter(bsbs, bc_voxels, c = 'darkgreen', s = 65, edgecolor = 'k', linewidths = 1.1)
plt.xlabel("bsbs")
plt.ylabel("bc_voxels")
sns.despine(top = True, right = True)
plt.tight_layout()
plt.savefig(path_results + 'fc_sc_bsbs_bc_voxels_R2.svg',
            dpi = 300)
print(spearmanr(bsbs, bc_voxels)) # 0.1596652883384558
plt.show()

#------------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------------

def spearman_perm_p(x, y, n_perm = 1000, seed = 0):
    """
    Permutation test for Spearman correlation.
    Shuffles y relative to x.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]

    r_obs, _ = spearmanr(x, y)

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_stats[i], _ = spearmanr(x, y_perm)

    p = pval_cal(r_obs, perm_stats, n_perm)
    return r_obs, p

r, p = spearman_perm_p(bsctx, bsbs, n_perm=1000, seed=1)
print("Spearman(bsctx, bsbs): r =", r, "perm p =", p) 
# Spearman(bsctx, bsbs): r = 0.2801378079916331 perm p = 0.023976023976023976

r, p = spearman_perm_p(bsctx, bc_voxels, n_perm=1000, seed=1)
print("Spearman(bsctx, voxels): r =", r, "perm p =", p)
# Spearman(bsctx, voxels): r = 0.3855039938668959 perm p = 0.004995004995004995

r, p = spearman_perm_p(bsbs, bc_voxels, n_perm=1000, seed=1)
print("Spearman(bsbs, voxels): r =", r, "perm p =", p)
# Spearman(bsbs, voxels): r = 0.1596652883384558 perm p = 0.22077922077922077

#------------------------------------------------------------------------------
'''
def partial_spearman_perm_p(x, y, z, n_perm=1000, seed=0):
    """
    Partial Spearman correlation between x and y controlling for z,
    with permutation p-value using Freedman–Lane:
      1) rank-transform x,y,z
      2) regress x~z and y~z -> residuals rx, ry
      3) permute ry and correlate rx with permuted ry
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]; y = y[mask]; z = z[mask]

    # rank-transform (Spearman)
    xr = rankdata(x)
    yr = rankdata(y)
    zr = rankdata(z).reshape(-1, 1)

    # regress out z
    lr = LinearRegression()
    lr.fit(zr, xr); rx = xr - lr.predict(zr)
    lr.fit(zr, yr); ry = yr - lr.predict(zr)

    # observed partial Spearman = Pearson corr on residual ranks
    r_obs = np.corrcoef(rx, ry)[0, 1]

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        ry_perm = rng.permutation(ry)
        perm_stats[i] = np.corrcoef(rx, ry_perm)[0, 1]

    p = (np.sum(perm_stats >= r_obs) + 1) / (n_perm + 1)

    return r_obs, p

r_partial, p_partial = partial_spearman_perm_p(bsctx, bsbs, bc_voxels, n_perm=20000, seed=2)
print("Partial Spearman(bsctx, bsbs | voxels): r =", r_partial, "perm p =", p_partial)
'''
#------------------------------------------------------------------------------
# END