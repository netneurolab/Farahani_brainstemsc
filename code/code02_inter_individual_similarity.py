"""

Inter-individual similarity:

cortex-cortex  vs  brainstem-cortex 
Mean difference: 0.1444
t-statistic: 48.2614
Classical t-test p = 2.781e-153
Permutation t-test p = 0.000999

cortex-cortex  vs  brainstem-brainstem 
Mean difference: -0.0315
t-statistic: -7.3166
Classical t-test p = 2.706e-12
Permutation t-test p = 0.000999

brainstem-cortex  vs  brainstem-brainstem 
Mean difference: -0.1758
t-statistic: -41.9586
Classical t-test p = 1.85e-118
Permutation t-test p = 0.000999

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import spearmanr
from globals import path_results, n_brainstem
from palettable.colorbrewer.sequential import PuBuGn_9

#------------------------------------------------------------------------------
# Load SC matrix of participants (N = 19)
#------------------------------------------------------------------------------

sc_subj = np.load(path_results + 'sc_subj.npy')
n_subj = 19

for i in range(n_subj): # make diagonal elements NaN
    np.fill_diagonal(sc_subj[i,:,:], np.nan)

#------------------------------------------------------------------------------
# Similarity of SC across subjects - show heatmaps
#------------------------------------------------------------------------------

similarity_bs_bs   = np.zeros((n_subj, n_subj)) # brainstem-brainstem
similarity_ctx_bs  = np.zeros((n_subj, n_subj)) # brainstem-cortex
similarity_ctx_cts = np.zeros((n_subj, n_subj)) # cortex-cortex

for i in range(n_subj):
    for j in range(n_subj):
        if i != j:
            similarity_bs_bs[i,j] = spearmanr(sc_subj[i,:n_brainstem,:n_brainstem].flatten(),
                                              sc_subj[j,:n_brainstem,:n_brainstem].flatten(),
                                              nan_policy = 'omit')[0]

            similarity_ctx_bs[i,j] = spearmanr(sc_subj[i,n_brainstem:,:n_brainstem].flatten(),
                                               sc_subj[j,n_brainstem:,:n_brainstem].flatten())[0]

            similarity_ctx_cts[i,j] = spearmanr(sc_subj[i,n_brainstem:,n_brainstem:].flatten(),
                                                sc_subj[j,n_brainstem:,n_brainstem:].flatten(),
                                                nan_policy = 'omit')[0]
            print(i)

fig, axes = plt.subplots(1, 3, figsize = (12, 4))
titles = [
    "brainstem-brainstem similarity",
    "brainstem-cortex similarity",
    "cortex-cortex similarity"]

mats = [
    similarity_bs_bs,
    similarity_ctx_bs,
    similarity_ctx_cts]

for ax, mat, title in zip(axes, mats, titles):
    im = ax.imshow(mat, vmin = 0, vmax = 1, cmap = PuBuGn_9.mpl_colormap)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(path_results + 'similarity_sc_across_people.png', format = 'png')
plt.show()

#------------------------------------------------------------------------------
# Similarity of SC across subjects - show jitter bar
#------------------------------------------------------------------------------

vals_ctx_ctx = similarity_ctx_cts[np.triu_indices(n_subj, k = 1)]
vals_ctx_bs  = similarity_ctx_bs[np.triu_indices(n_subj, k = 1)]
vals_bs_bs   = similarity_bs_bs[np.triu_indices(n_subj, k = 1)]

groups = [vals_bs_bs, vals_ctx_bs, vals_ctx_ctx]
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(3)
bar_width = 0.6
for i, g in enumerate(groups):
    jitter = (np.random.rand(len(g)) - 0.5) * bar_width * 0.8 
    ax.scatter(x[i] + jitter, g, color = 'silver', alpha = 0.6, s = 10)

ax.set_xticks(x)
ax.set_xticklabels(titles)
ax.set_ylabel("Spearman similarity")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_results + 'similarity_sc_across_people_jitter.svg', format = 'svg')
plt.show()

# Print the avergae spearman correlation per each compartment
print('bs-bs')
print(np.mean(vals_bs_bs))

print('ctx-ctx')
print(np.mean(vals_ctx_ctx))

print('xtc-bs')
print(np.mean(vals_ctx_bs))

#------------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------------

groups = {
    "cortex-cortex":       vals_ctx_ctx,
    "brainstem-cortex":    vals_ctx_bs,
    "brainstem-brainstem": vals_bs_bs}

pairs = [
    ("cortex-cortex",    "brainstem-cortex"),
    ("cortex-cortex",    "brainstem-brainstem"),
    ("brainstem-cortex", "brainstem-brainstem")]

# Permutation test function
def permutation_test(a, b, n_perm = 1000):
    t_obs = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    diff_perm = np.zeros(n_perm)
    for i in range(n_perm):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff_perm[i] = np.mean(perm_a) - np.mean(perm_b)
    p_perm = pval_cal(t_obs, diff_perm , n_perm)
    return diff_perm, p_perm

results = []
for g1, g2 in pairs:
    a = groups[g1]
    b = groups[g2]

    # Classical t-test
    t_val, p_t = ttest_ind(a, b, equal_var = False)

    # Permutation t-test
    distribution, p_perm = permutation_test(a, b)
    results.append((g1, g2, np.mean(a), np.mean(b), t_val, p_t, p_perm))

for r in results:
    g1, g2, m1, m2, t_val, p_t, p_perm = r
    print(f"\n {g1}  vs  {g2} ")
    print(f"Mean difference: {m1 - m2:.4f}")
    print(f"t-statistic: {t_val:.4f}")
    print(f"Classical t-test p = {p_t:.4g}")
    print(f"Permutation t-test p = {p_perm:.4g}")

#------------------------------------------------------------------------------
# END