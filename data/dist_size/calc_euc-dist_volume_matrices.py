#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:39:13 2025

@author: fh002
"""

import sys, numpy as np, pandas as pd

f = '/space/segesta/4/users/fabrice/data/1.ppmi/2.MRI/labels/lin22fabrice_revised.csv'
df = pd.read_csv(f)

roi    = df['roi_list1'].values
coords = df[['x','y','z']].to_numpy(float)
vol    = df['volume'].to_numpy(float)

# Euclidean distance
dist = np.sqrt(((coords[:,None] - coords[None,:])**2).sum(-1))

# sqrt(volume_i * volume_j)
volsqrt = np.sqrt(vol[:,None] * vol[None,:])

# DataFrames
dist_df    = pd.DataFrame(dist,    index=roi, columns=roi)
volsqrt_df = pd.DataFrame(volsqrt, index=roi, columns=roi)

# to save as .csv file
dist_df.to_csv("/space/segesta/4/users/fabrice/data/1.ppmi/2.MRI/labels/dist.csv")
volsqrt_df.to_csv("/space/segesta/4/users/fabrice/data/1.ppmi/2.MRI/labels/volsqrt.csv")
