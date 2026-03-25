"""

The saved file --> "consenses_degree_lenght_preserving_nulls.npy"
Size of the numpy array is 1000 by 458 by 458.
We used "match_length_degree_distribution" function to create the null networks.

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from netneurotools import networks
from globals import path_results, path_dist_size

#------------------------------------------------------------------------------
# Load consensus SC
#------------------------------------------------------------------------------

consenses_vis = np.load(path_results + 'consensus.npy')
np.fill_diagonal(consenses_vis, 0)

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
# Create consensus SC nulls
#------------------------------------------------------------------------------

nspins = 1000
null_corrs = np.zeros((nspins, 458, 458))

for s in range(nspins):
    null_corrs[s,:,:] = networks.match_length_degree_distribution(consenses_vis, dist)[1]
    print(s) # shows progress

np.save(path_results + 'consenses_degree_lenght_preserving_nulls.npy', null_corrs)

#------------------------------------------------------------------------------
# END