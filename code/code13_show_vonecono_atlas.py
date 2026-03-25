"""

Save on-Economo classes on cortical map

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
from functions import show_on_surface_and_save_cmap
from globals import path_results, path_atlas
from matplotlib.colors import ListedColormap

#------------------------------------------------------------------------------
# show brainstem is more connected to motor cortex
#------------------------------------------------------------------------------

my_colors = ["#015544", # 'primary motor'
             "#8c92cc", # 'association'
             "#c7bfd4", # 'association'
             "#5589ff", # 'primary/secondary sensory'
             "#76c2de", # 'primary sensory'
             "#de7eaf", # 'limbic'
             "#769490"] # 'insular'


cmap = ListedColormap(my_colors)

atlas_data = np.squeeze(scipy.io.loadmat(path_atlas + 'economo_Schaefer400.mat')['pdata'])

atlas_data = atlas_data

in_data = atlas_data.reshape(400, 1)

show_on_surface_and_save_cmap(
    in_data, 400, path_results, "economo_Schaefer400.png",
    cmap = cmap, color_range = (0.5, 7.5))

#------------------------------------------------------------------------------
# END