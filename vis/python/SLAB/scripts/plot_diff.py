import sys
sys.path.append("../../")
import os
import numpy as np
from athena_read import athdf
import matplotlib.pyplot as plt
import Constants 
c=Constants.Constants()
from cdl import helmholtz_decomposition

base_dir = '../../../../data/TDSC/'
run_dir = 'M10_B0.1_R2_D0.02_PR/'
base_dir += run_dir

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

# Read files
myfile=base_dir + "COLL.out1."+"00050"+".athdf"
d = athdf(myfile)
X,Y = np.meshgrid(d['x1f'], d['x2f'], indexing='ij')
vx_s,vy_s,vz_s, vx_c,vy_c,vz_c = helmholtz_decomposition(d)

# Plot beta
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
contour = ax.pcolormesh(X, Y, ((d['vel1']-vx_c-vx_s)[32,:,:]).T,cmap='plasma',shading='auto')
cb1=plt.colorbar(contour,extend='both',label=r'$\log_{10}\left( \beta \right)$ ')
# ax.axvline(x=d['x1f'][226], color='k', ls='--')
# ax.axvline(x=d['x1f'][286], color='k', ls='--')
ax.set_xlabel("x")
ax.set_ylabel("y")

out_dir = '../figs/' + run_dir
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + '/diffv.pdf', format='pdf', bbox_inches='tight')