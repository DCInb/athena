import sys
sys.path.append("../../../")
import os
import numpy as np
from athena_read import athdf
import matplotlib.pyplot as plt
import Constants 
c=Constants.Constants()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="HDF5 file to plot", required=True)
args = parser.parse_args()
myfile = args.file
print(f"Plotting file: {myfile}")

# get the run number from the file path
run_num = myfile.split('.')[-2]
run_dir = myfile.split('/')[-2]

import matplotlib as mpl

mpl.rcParams.update({
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.default": "regular",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.unicode_minus": True,
    
    # Log scale specific additions
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
})

d = athdf(myfile)

# --- choose slice indices ---
iz = 127   # z index for xy plane
ix = 256   # x index for yz plane

# ----------------------------
# coordinates
# cell centers are better for streamplot
# face coords are fine for pcolormesh
# ----------------------------
x1f, x2f, x3f = d['x1f'], d['x2f'], d['x3f']
x1v, x2v, x3v = d['x1v'], d['x2v'], d['x3v']

Xf_xy, Yf_xy = np.meshgrid(x1f, x2f, indexing='ij')
Yf_yz, Zf_yz = np.meshgrid(x2f, x3f, indexing='ij')

# streamplot grids
Xc_xy, Yc_xy = np.meshgrid(x1v, x2v, indexing='ij')
Yc_yz, Zc_yz = np.meshgrid(x2v, x3v, indexing='ij')

# ----------------------------
# scalar fields
# Athena arrays assumed shape = (nz, ny, nx)
# ----------------------------
rho = d['rho']
vel_mag = np.sqrt(d['vel1']**2 + d['vel2']**2 + d['vel3']**2)
B_mag   = np.sqrt(d['Bcc1']**2 + d['Bcc2']**2 + d['Bcc3']**2)

fields = [
    (np.log10(rho),     r'$\log_{10}(\rho)$',   'Density',        'viridis'),
    (np.log10(vel_mag), r'$\log_{10}(|v|)$',    'Velocity',       'plasma'),
    (np.log10(B_mag),   r'$\log_{10}(|B|)$',    'Magnetic field', 'inferno'),
]

# ----------------------------
# vector fields for streamlines
# xy plane at z = iz
# yz plane at x = ix
# transpose to match meshgrid(indexing='ij') plotting orientation
# ----------------------------

# velocity components
vx_xy = d['vel1'][iz, :, :].T   # (nx, ny)
vy_xy = d['vel2'][iz, :, :].T

vy_yz = d['vel2'][:, :, ix].T   # (ny, nz)
vz_yz = d['vel3'][:, :, ix].T

# magnetic components
Bx_xy = d['Bcc1'][iz, :, :].T
By_xy = d['Bcc2'][iz, :, :].T

By_yz = d['Bcc2'][:, :, ix].T
Bz_yz = d['Bcc3'][:, :, ix].T

fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
# store one mappable per column
mappables = []

for col, (field, cbar_label, title, cmap) in enumerate(fields):
    # ----------------------------
    # top row: xy plane
    # ----------------------------
    ax = axes[0, col]

    pcm = ax.pcolormesh(
        Xf_xy, Yf_xy, field[iz, :, :].T,
        cmap=cmap,
        shading='auto'
    )

    # velocity streamlines
    if col==1:
        ax.streamplot(
            x1v, x2v,
            vx_xy.T, vy_xy.T,
            color='white',
            density=1.0,
            linewidth=0.7,
            arrowsize=0.8
        )

    # magnetic field streamlines
    if col==2:
        ax.streamplot(
            x1v, x2v,
            Bx_xy.T, By_xy.T,
            color='cyan',
            density=1.0,
            linewidth=0.7,
            arrowsize=0.8
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    # ----------------------------
    # bottom row: yz plane
    # ----------------------------
    ax = axes[1, col]

    pcm2 = ax.pcolormesh(
        Yf_yz, Zf_yz, field[:, :, ix].T,
        cmap=cmap,
        shading='auto'
    )
    # save for shared colorbar
    mappables.append((pcm2, cbar_label))

    # velocity streamlines
    if col==1:
        ax.streamplot(
            x2v, x3v,
            vy_yz.T, vz_yz.T,
            color='white',
            density=1.0,
            linewidth=0.7,
        arrowsize=0.8
    )

    # magnetic field streamlines
    if col==2:
        ax.streamplot(
            x2v, x3v,
            By_yz.T, Bz_yz.T,
            color='cyan',
            density=1.0,
            linewidth=0.7,
        arrowsize=0.8
    )

    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

# -------- shared colorbars (one per column) --------
for col, (pcm, cbar_label) in enumerate(mappables):
    fig.colorbar(
        pcm,
        ax=axes[:, col],
        extend='both',
        label=cbar_label,
        location='bottom',
        pad=0.02
    )

out_dir = 'figs/' + run_dir + '/slices'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + '/slices_fields_'+str(run_num)+'.png', dpi=300, format='png', bbox_inches='tight')
