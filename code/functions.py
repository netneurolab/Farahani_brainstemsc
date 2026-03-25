import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
from surfplot import Plot
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from scipy.stats import spearmanr
from neuromaps.nulls import spins
from collections.abc import Iterable 
from matplotlib.colors import Normalize
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
from palettable.colorbrewer.sequential import PuBuGn_9
from sklearn.utils.validation import check_random_state
from globals import path_wb_command, path_results, path_fc, path_atlas

#------------------------------------------------------------------------------

def show_on_surface_and_save_cmap(in_data, nnodes, path_results, fig_name,
                                 cmap, color_range = None,
                                 zoom = 1.2, size = (900, 450)):
    """
    in_data: (nnodes, 1) values per parcel
    cmap: matplotlib colormap (can be ListedColormap)
    color_range: (vmin, vmax). If None, computed from data.
    """
    surfaces = fetch_fslr()
    lh, rh = surfaces['midthickness']
    lh_parc, rh_parc = load_parcellation('schaefer', scale = nnodes)
    in_vec = np.asarray(in_data).reshape(nnodes,)
    regions_lh = np.zeros_like(lh_parc, dtype = float)
    regions_rh = np.zeros_like(rh_parc, dtype = float)

    for i in range(1, nnodes + 1):
        regions_lh[lh_parc == i] = in_vec[i - 1]
        regions_rh[rh_parc == i] = in_vec[i - 1]

    if color_range is None:
        vmin = np.nanmin(in_vec)
        vmax = np.nanmax(in_vec)
        color_range = (vmin, vmax)

    # left hemi
    p = Plot(lh, size = size, zoom = zoom, layout = 'row')
    p.add_layer(regions_lh, cmap = cmap, color_range = color_range)
    p.add_layer(regions_lh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.savefig(os.path.join(path_results, 'lh.' + fig_name), dpi = 300)

    # right hemi
    p = Plot(rh, size = size, zoom = zoom, layout = 'row')
    p.add_layer(regions_rh, cmap = cmap, color_range = color_range)
    p.add_layer(regions_rh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.savefig(os.path.join(path_results, 'rh.' + fig_name), dpi = 300)

#------------------------------------------------------------------------------

def show_on_surface_and_save(in_data, nnodes, rangeLow, rangehigh,
                             path_results, fig_name):
    """
    Show the data on the surface and also save the created figures as a png file
    """

    color_range = (rangeLow,  rangehigh)
    surfaces = fetch_fslr()
    lh, rh = surfaces['midthickness']
    lh_parc, rh_parc = load_parcellation('schaefer', scale = nnodes)
    regions_lh = np.zeros_like(lh_parc)
    regions_rh = np.zeros_like(rh_parc)

    for i in range(1, nnodes + 1):
        regions_lh = np.where(np.isin(lh_parc, i),
                              in_data[i-1, 0], 0) + regions_lh
        regions_rh = np.where(np.isin(rh_parc, i),
                              in_data[i-1, 0], 0) + regions_rh

    p = Plot(lh, size = (nnodes, int(nnodes/2)), zoom = 1.2, layout = 'row') # left hemisphere
    p.add_layer(regions_lh, cmap = PuBuGn_9.mpl_colormap, color_range = color_range)
    p.add_layer(regions_lh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.show()
    fig.savefig(os.path.join(path_results,'lh.' + fig_name), dpi = 300) # Save left hemisphere figure

    p = Plot(rh, size = (nnodes, int(nnodes/2)), zoom = 1.2, layout = 'row') # right hemisphere
    p.add_layer(regions_rh, cmap = PuBuGn_9.mpl_colormap, color_range = color_range)
    p.add_layer(regions_rh, cmap = 'gray', as_outline = True, cbar = False)
    fig = p.build()
    fig.show()
    fig.savefig(os.path.join(path_results, 'rh.' + fig_name), dpi = 300) # Save right hemisphere figure

#------------------------------------------------------------------------------

def vasa_null_Schaefer(nspins):
    # Info related to spin tests
    coords = np.genfromtxt(path_atlas + 'Schaefer_400.txt')
    coords = coords[:, -3:]
    nnodes = len(coords)
    hemiid = np.zeros((nnodes,))
    hemiid[:int(nnodes/2)] = 1
    spins_ = spins.gen_spinsamples(coords,
                            hemiid,
                            n_rotate = nspins,
                            seed = 12345,
                            method = 'vasa')
    return spins_

#------------------------------------------------------------------------------

def pval_cal(rho_actual, null_dis, num_spins):
    """
    Calculate p-value - non parametric
    """
    p_value = (1 + np.count_nonzero(abs((null_dis - np.mean(null_dis))) > abs((rho_actual - np.mean(null_dis))))) / (num_spins + 1)
    return(p_value)

#------------------------------------------------------------------------------

def pval_cal_1side(rho_actual, null_dis, num_spins):
    """
    Calculate p-value - non parametric
    """
    p_value = (1 + np.count_nonzero((null_dis - np.mean(null_dis)) > (rho_actual - np.mean(null_dis)))) / (num_spins + 1)
    return(p_value)

#------------------------------------------------------------------------------

def plot_spearman(x, y, outname, labels = None):
    """
    Scatter plot with Spearman correlation coefficient and optional point labels.

    Parameters:
    - x, y: Arrays of values to correlate.
    - outname: Filename for saving the plot (saved in path_figures).
    - labels: Optional list or array of region names for labeling each point.
    """
    rho, pval = spearmanr(x, y, nan_policy = 'omit')
    plt.figure(figsize = (6, 6))
    plt.scatter(x, y, color = 'silver')
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(x[i], y[i], str(label), fontsize = 6, alpha = 0.7)
    plt.text(0.05, 0.95, f"Spearman r = {rho:.2f}\np = {pval:.2e}",
             transform = plt.gca().transAxes,
             fontsize = 12,
             verticalalignment = 'top')

    sns.despine(top = True, right = True)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------

def plot_brainstem_with_names(angle):
    label_info = pd.read_csv(path_fc + 'region_info_Schaefer400.csv')
    fig, ax = plt.subplots(figsize = (10, 8),
                           subplot_kw = dict(projection = '3d'))

    coords = label_info.query('structure == "brainstem"')[['x', 'y', 'z']].reset_index(drop = True)
    filtered_list = np.array(label_info.query('structure == "brainstem"')['labels'])
    # Plot each brainstem region
    for i in range(len(coords)):
        if np.isnan(coords.loc[i, 'x']):
            continue
        ax.scatter(coords.loc[i, 'x'], coords.loc[i, 'y'], coords.loc[i, 'z'],
                   s = 10**1.8,
                   alpha = 0.9,
                   edgecolors = 'k')
        ax.text(coords.loc[i, 'x'], coords.loc[i, 'y'], coords.loc[i, 'z'],
                filtered_list[i], # region label
                size = 8, zorder = 1, color = 'k')
    scaling = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ax.view_init(0, angle)
    ax.axis('off')
    ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------

def plot_brainstem_with_names_size(
    angle,
    size_col='nvoxels',
    label_col='labels',
    s_range=(40, 200),     # min/max marker area for scatter (points^2)
    scale='sqrt'           # 'linear' | 'sqrt' | 'log'
):
    # Load and filter (case-insensitive)
    df = pd.read_csv(path_fc + 'region_info_Schaefer400.csv')
    brain = df[df['structure'].str.lower() == 'brainstem'].copy()

    # Basic checks
    for c in ('x','y','z', size_col, label_col):
        if c not in brain.columns:
            raise ValueError(f"Missing column '{c}' in CSV.")
    brain = brain.dropna(subset=['x','y','z', size_col])

    # Size scaling from nvoxels
    vals = brain[size_col].to_numpy().astype(float)

    if scale == 'sqrt':
        v = np.sqrt(np.clip(vals, a_min=0, a_max=None))
    elif scale == 'log':
        v = np.log1p(np.clip(vals, a_min=0, a_max=None))
    else: # 'linear'
        v = vals

    vmin, vmax = np.min(v), np.max(v)
    if np.isclose(vmax, vmin):
        norm = np.ones_like(v) # all same size if no variance
    else:
        norm = (v - vmin) / (vmax - vmin)

    s_min, s_max = s_range
    sizes = s_min + norm * (s_max - s_min) # matplotlib 's' is area, not radius

    fig, ax = plt.subplots(figsize = (10, 8), subplot_kw = dict(projection = '3d'))
    ax.scatter(brain['x'].values, brain['y'].values, brain['z'].values,
               s = sizes, alpha = 0.9, edgecolors = 'k',
               linewidths = 0.5, marker = 'o')

    for x, y, z, lab in brain[['x','y','z', label_col]].itertuples(index = False, name = None):
        ax.text(x, y, z, str(lab), size = 8, zorder = 2, color = 'k')

    pad = 1.0
    xmin, xmax = brain['x'].min()-pad, brain['x'].max()+pad
    ymin, ymax = brain['y'].min()-pad, brain['y'].max()+pad
    zmin, zmax = brain['z'].min()-pad, brain['z'].max()+pad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))

    ax.view_init(elev = 0, azim = float(angle))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path_results+'temp.svg')
    plt.show()

#------------------------------------------------------------------------------

def plot_network(name, A, coords, edge_scores, node_scores,
                 edge_cmap = "Greys", node_cmap = "viridis",
                 edge_alpha = 0.25, node_alpha = 1,
                 edge_vmin = None, edge_vmax = None, node_vmin = None,
                 node_vmax = None, nodes_color = 'black', edges_color = 'black',
                 linewidth = 0.25,
                 s = 100,                    # default scalar size (fallback)
                 node_sizes = None,          # NEW: raw region sizes
                 size_vmin = None, size_vmax = None,  # NEW: min/max of node_sizes
                 size_range = (20, 300),     # NEW: marker size range in points^2
                 view_edge = True, figsize = None,
                 views = None, views_orientation = 'vertical',
                 threshold = False, valth = 0):

    _views = dict(sagittal = (0, 180), sag = (0, 180),
                  axial = (90, 180), ax = (90, 180),
                  coronal = (0, 90), cor = (0, 90))

    # Process the 'views' parameter
    if views is None:
        views = ['sagittal', 'axial']
    elif isinstance(views, str):
        views = [views.lower()]
    elif isinstance(views, Iterable):
        views = [v.lower() for v in views]
    else:
        raise ValueError("views must be a string or a list of strings.")

    # Determine subplot layout
    n_views = len(views)
    if views_orientation == 'vertical':
        ncols, nrows = 1, n_views
    elif views_orientation == 'horizontal':
        ncols, nrows = n_views, 1
    else:
        raise ValueError("views_orientation must be 'vertical' or 'horizontal'")

    figsize = (ncols * 10, nrows * 6)
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                             figsize = figsize,
                             subplot_kw=dict(projection = '3d'))
    if n_views == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine edges to plot
    if threshold:
        edges = np.where(A > valth)
    else:
        edges = np.where(A)

    # Normalize edge scores if provided
    if edge_scores is not None:
        edge_norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_cmap_func = cm.get_cmap(edge_cmap)
        edge_colors_all = edge_cmap_func(edge_norm(edge_scores[edges]))
    else:
        edge_colors_all = [edges_color] * len(edges[0])

    # Normalize node scores if provided
    if node_scores is not None:
        node_norm = mpl.colors.Normalize(vmin=node_vmin, vmax=node_vmax)
        node_cmap_func = cm.get_cmap(node_cmap)
        node_colors = node_cmap_func(node_norm(node_scores))
    else:
        node_colors = nodes_color

    # --- NEW: compute node sizes from region sizes ---
    if node_sizes is not None:
        node_sizes = np.asarray(node_sizes)
        if size_vmin is None:
            size_vmin = np.nanmin(node_sizes)
        if size_vmax is None:
            size_vmax = np.nanmax(node_sizes)
        if size_vmax == size_vmin:
            # avoid division by zero – fall back to constant
            s_array = np.full_like(node_sizes, np.mean(size_range), dtype=float)
        else:
            # linear scaling to size_range
            s_array = (node_sizes - size_vmin) / (size_vmax - size_vmin)
            s_array = size_range[0] + s_array * (size_range[1] - size_range[0])
    else:
        s_array = s  # scalar or array directly provided
    # -------------------------------------------------

    # Plot for each view
    for ax, view in zip(axes, views):
        elev, azim = _views.get(view, (0, 0))
        ax.view_init(elev=elev, azim=azim)

        # Plot edges
        if view_edge:
            for idx, (i, j) in enumerate(zip(edges[0], edges[1])):
                x_coords = [coords[i, 0], coords[j, 0]]
                y_coords = [coords[i, 1], coords[j, 1]]
                z_coords = [coords[i, 2], coords[j, 2]]
                ax.plot(x_coords, y_coords, z_coords,
                        color=edge_colors_all[idx],
                        linewidth=linewidth,
                        alpha=edge_alpha, zorder=0)

        # Plot nodes
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=node_colors,
                   cmap=node_cmap,
                   vmin=node_vmin, vmax=node_vmax,
                   alpha=node_alpha,
                   s=s_array,           # <- size now depends on region size
                   edgecolors='none', zorder=1)

        scaling = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))
        ax.axis('off')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.savefig(path_results + name + '.svg')
    return fig, ax

#------------------------------------------------------------------------------

def sequential_color1(N=100, return_palette=False, n_colors=8):
    """
    Generate a sequential green colormap.

    Parameters
    ----------
    N : int, optional
        Number of colors in the colormap. Default is 100.
    return_palette : bool, optional
        If True, return a seaborn color palette instead of a colormap. Default is False.
    n_colors : int, optional
        Number of colors in the palette. Only applicable if return_palette is True. Default is 8.

    Returns
    -------
    colormap or color palette
        A matplotlib colormap or seaborn color palette.

    Examples
    --------
    Generate a sequential blue colormap with 50 colors:
    >>> cmap = sequential_green(N=50)

    Generate a seaborn color palette with 5 colors:
    >>> palette = sequential_green(return_palette=True, n_colors=5)
    """
    clist = ["666a86", "788aa3","ffffff","e3c9c9","b17b7b"]
    hex = [f'#{c}' for c in clist]
    rgb = list(map(mpc.to_rgb, hex))
    if return_palette:
        return sns.color_palette(rgb, n_colors=n_colors)
    else:
        return mpc.LinearSegmentedColormap.from_list('custom', rgb, N=N)

#------------------------------------------------------------------------------

def save_gifti(file, file_name):
    import nibabel as nib
    da=nib.gifti.GiftiDataArray(file,datatype='NIFTI_TYPE_FLOAT32')
    img=nib.GiftiImage(darrays=[da])
    nib.save(img,(file_name+'.func.gii'))  

#------------------------------------------------------------------------------

def parcel2fsLR(atlas, data_parcelwise, hem):
    """
    Generate 32492 valuse for vertices from a parcelwise data
    """
    if (np.size(data_parcelwise.shape) != 1):
        results = np.zeros((32492,
                            int(np.size(data_parcelwise, axis = 1))))
    else:
        results = np.zeros((32492,))
    if hem in ['l', 'L']:
        atlas_hem = atlas[0: 32492]
    if hem in ['r', 'R']:
        atlas_hem = atlas[32492:]

    unique_labels_hem = np.sort(np.unique(atlas_hem))
    for count, x in enumerate(unique_labels_hem[1:]):
        results[atlas_hem == x,:] = data_parcelwise[count, :]
    return results

#------------------------------------------------------------------------------

def randmio_und(W, itr):
    """
    Optimized version of randmio_und.

    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    This function is significantly faster if numba is enabled, because
    the main overhead is `np.random.randint`, see `here <https://stackoverflow.com/questions/58124646/why-in-python-is-random-randint-so-much-slower-than-random-random>`_

    Parameters
    ----------
    W : (N, N) array-like
        Undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    W : (N, N) array-like
        Randomized network
    eff : int
        number of actual rewirings carried out
    """  # noqa: E501
    W = W.copy()
    n = len(W)
    i, j = np.where(np.triu(W > 0, 1))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for _ in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k), np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a, b = i[e1], j[e1]
                c, d = i[e2], j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # flip edge c-d with 50% probability
            # to explore all potential rewirings
            if np.random.random() > .5:
                i[e2], j[e2] = d, c
                c, d = d, c

            if not (W[a, d] or W[c, b]):
                W[a, d] = W[a, b]
                W[a, b] = 0
                W[d, a] = W[b, a]
                W[b, a] = 0
                W[c, b] = W[c, d]
                W[c, d] = 0
                W[b, c] = W[d, c]
                W[d, c] = 0

                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return W, eff

#------------------------------------------------------------------------------

def cifti2gifti(input_file_name, input_file_path):
    """
    Convert cifti file into gifti using the workbench commands
    """
    command = path_wb_command + 'wb_command -cifti-separate ' + \
    input_file_path + input_file_name + '.dscalar.nii' + ' COLUMN -metric CORTEX_LEFT ' + \
    input_file_path + 'lh.' + input_file_name + '.func.gii'
    os.system(command)
    command = path_wb_command + 'wb_command -cifti-separate ' + \
    input_file_path + input_file_name + '.dscalar.nii' + ' COLUMN -metric CORTEX_RIGHT ' + \
    input_file_path + 'rh.' + input_file_name + '.func.gii'
    os.system(command)

#------------------------------------------------------------------------------

def convert2ciftidense(path_out , cifti_out,
                  path_metrics, metric_left, metric_right):
    """
    Convert into a cifti file
    """
    command = path_wb_command + 'wb_command -cifti-create-dense-scalar ' + \
    path_out + cifti_out + '.dscalar.nii ' + \
    '-left-metric ' + path_metrics + metric_left + '.func.gii ' +\
    '-right-metric ' + path_metrics + metric_right + '.func.gii '
    os.system(command)

#------------------------------------------------------------------------------

def save_and_convert_to_cifti(atlas, data, nnodes, file_name, path_results):
    """
    Saves the left and right hemisphere GIFTI files, converts them to CIFTI format, 
    and cleans up the intermediate GIFTI files.

    Parameters:
    - atlas: The atlas data to be used for parcellation.
    - data: The data array that needs to be parcellated and saved.
    - nnodes: The number of nodes in the atlas.
    - subtype: The subtype of the data, used in naming the files.
    - path_wb_command: The path to the wb_command tool.
    - path_results: The directory where the results will be saved.
    """

    hemisphere_keys = ['L', 'R']
    hemi_data_splits = [data[:int(nnodes/2) ,:].reshape(int(nnodes/2), len(data.T)),
                        data[int(nnodes/2):,:].reshape(int(nnodes/2), len(data.T))]

    # Generate and save GIFTI files for each hemisphere
    for hemi_key, hemi_data in zip(hemisphere_keys, hemi_data_splits):
        hemi_file_name = f'{hemi_key.lower()}h.{file_name}'
        hemi_gifti = parcel2fsLR(atlas, hemi_data, hemi_key)
        save_gifti(hemi_gifti, os.path.join(path_results, hemi_file_name))

    # Convert to CIFTI
    convert2ciftidense(path_results, file_name, path_results, 'lh.' + file_name, 'rh.' + file_name)

    # Cleanup intermediate GIFTI files
    for hemi_key in hemisphere_keys:
        hemi_file_path = os.path.join(path_results, f'{hemi_key.lower()}h.{file_name}.func.gii')
        if os.path.exists(hemi_file_path):
            os.remove(hemi_file_path)

#------------------------------------------------------------------------------

def plot_brainstem(in_data, name, degree, min_value, max_value):
    info = pd.read_csv(path_fc+ '/region_info_Schaefer400.csv',
                       index_col=0)
    filtered_list = info.query("structure == 'brainstem'").labels
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='3d'))
    coords = info.query('structure == "brainstem"')[['x', 'y', 'z']].reset_index(drop=True)
    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap = cm.get_cmap('coolwarm') 
    colors = cmap(norm(in_data))
    
    # Plot each brainstem region
    for i in range(len(coords)):
        if np.isnan(coords.loc[i, 'x']):
            continue
        ax.scatter(coords.loc[i, 'x'], coords.loc[i, 'y'], coords.loc[i, 'z'],
                   s=10**1.8,
                   c=colors[i].reshape(1, -1),  # Color from colormap
                   alpha=0.9,
                   edgecolors='k')
        
        ax.text(coords.loc[i, 'x'], coords.loc[i, 'y'], coords.loc[i, 'z'],
                filtered_list[i],  # region label
                size=8, zorder=1, color='k')
    
    # Axis and view
    scaling = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ax.view_init(0, degree)
    ax.axis('off')
    ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))
    
    # Add a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6, ax=ax, pad=0.1)
    cbar.set_label('Atrophy (inverted)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# nulls
#------------------------------------------------------------------------------

def match_length_degree_distribution(W, D, nbins = 10, nswap = 1000,
                                     replacement = False, weighted = True,
                                     seed=None):
    """
    Generate degree- and edge length-preserving surrogate connectomes.

    Parameters
    ----------
    W : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    D : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20
        Default = 1000.
    replacement : bool, optional
        if True all the edges are available for swapping. Default= False
    weighted : bool, optional
        Whether to return weighted rewired connectivity matrix. Default = True
    seed : float, optional
        Random seed. Default = None

    Returns
    -------
    newB : (N, N) array-like
        binary rewired matrix
    newW: (N, N) array-like
        weighted rewired matrix. Returns matrix of zeros if weighted=False.
    nr : int
        number of successful rewires

    Notes
    -----
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    """
    rs = check_random_state(seed)
    N = len(W)
    # divide the distances by lengths
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N, N))
    for n in range(nbins):
        i, j = np.where(np.logical_and(bins[n] <= D, D < bins[n + 1]))
        L[i, j] = n + 1

    # binarized connectivity
    B = (W != 0).astype(np.int_)

    # existing edges (only upper triangular cause it's symmetric)
    cn_x, cn_y = np.where(np.triu((B != 0) * B, k=1))

    tries = 0
    nr = 0
    newB = np.copy(B)

    while ((len(cn_x) >= 2) & (nr < nswap)):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r], cn_y[r]
        tries += 1

        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if len(np.where(index)[0]) == 0:
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)

        else:
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            index = (L[n_x, n_y] == L[n_x, ops1_x]) & (
                L[ops1_x, ops1_y] == L[n_y, ops1_y])
            if len(np.where(index)[0]) == 0:
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)

            else:
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [(newB[min(n_x, ops2_x[i])][max(n_x, ops2_x[i])] == 0) & (newB[min(n_y, ops2_y[i])][max(n_y, ops2_y[i])] == 0)
                         for i in range(len(ops2_x))]
                if (len(np.where(index)[0]) == 0):
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)

                else:
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]

                    # choose randomly one edge from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]

                    # Disconnect the existing edges
                    newB[n_x, n_y] = 0
                    newB[nn_x, nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x, nn_x), max(n_x, nn_x)] = 1
                    newB[min(n_y, nn_y), max(n_y, nn_y)] = 1
                    # one successfull rewire!
                    nr += 1

                    # rewire with replacement
                    if replacement:
                        cn_x[r], cn_y[r] = min(n_x, nn_x), max(n_x, nn_x)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index], cn_y[index] = min(n_y, nn_y), max(n_y, nn_y)
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, r)
                        cn_y = np.delete(cn_y, r)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)

    if nr < nswap:
        print(f"I didn't finish, out of rewirable edges: {len(cn_x)}")

    i, j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j, i] = newB[i, j]

    # check the number of edges is preserved
    if len(np.where(B != 0)[0]) != len(np.where(newB != 0)[0]):
        print(
            f"ERROR --- number of edges changed, \
            B:{len(np.where(B!=0)[0])}, newB:{len(np.where(newB!=0)[0])}")
    # check that the degree of the nodes it's the same
    for i in range(N):
        if np.sum(B[i]) != np.sum(newB[i]):
            print(
                f"ERROR --- node {i} changed k by: \
                {np.sum(B[i]) - np.sum(newB[i])}")

    newW = np.zeros((N, N))
    if weighted:
        # Reassign the weights
        mask = np.triu(B != 0, k=1)
        inids = D[mask]
        iniws = W[mask]
        inids_index = np.argsort(inids)
        # Weights from the shortest to largest edges
        iniws = iniws[inids_index]
        mask = np.triu(newB != 0, k=1)
        finds = D[mask]
        i, j = np.where(mask)
        # Sort the new edges from the shortest to the largest
        finds_index = np.argsort(finds)
        i_sort = i[finds_index]
        j_sort = j[finds_index]
        # Assign the initial sorted weights
        newW[i_sort, j_sort] = iniws
        # Make it symmetrical
        newW[j_sort, i_sort] = iniws

    return newB, newW, nr

#------------------------------------------------------------------------------

def randmio_und(W, itr):
    """
    Optimized version of randmio_und.

    This function randomizes an undirected network, while preserving the
    degree distribution. The function does not preserve the strength
    distribution in weighted networks.

    This function is significantly faster if numba is enabled, because
    the main overhead is `np.random.randint`, see `here <https://stackoverflow.com/questions/58124646/why-in-python-is-random-randint-so-much-slower-than-random-random>`_

    Parameters
    ----------
    W : (N, N) array-like
        Undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.

    Returns
    -------
    W : (N, N) array-like
        Randomized network
    eff : int
        number of actual rewirings carried out
    """  # noqa: E501
    W = W.copy()
    n = len(W)
    i, j = np.where(np.triu(W > 0, 1))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for _ in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k), np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a, b = i[e1], j[e1]
                c, d = i[e2], j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            # flip edge c-d with 50% probability
            # to explore all potential rewirings
            if np.random.random() > .5:
                i[e2], j[e2] = d, c
                c, d = d, c

            if not (W[a, d] or W[c, b]):
                W[a, d] = W[a, b]
                W[a, b] = 0
                W[d, a] = W[b, a]
                W[b, a] = 0
                W[c, b] = W[c, d]
                W[c, d] = 0
                W[b, c] = W[d, c]
                W[d, c] = 0

                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return W, eff

#------------------------------------------------------------------------------
# END
