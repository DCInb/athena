import numpy as np
import matplotlib.pyplot as plt
import os

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

run_dir = 'M5_B0.1_R2_D0.02_PR'
cdl_5 = np.loadtxt('../../dat/' + run_dir + '.dat')
run_dir = 'M15_B0.1_R2_D0.02_PR'
cdl_15 = np.loadtxt('../../dat/dat_weighted/' + run_dir + '.dat')
run_dir = 'M20_B0.1_R2_D0.02_PR'
cdl_20= np.loadtxt('../../dat/dat_weighted/' + run_dir + '.dat')
run_dir = 'M10_B0.1_R2_D0.02_PR'
cdl_10= np.loadtxt('../../dat/dat_weighted/' + run_dir + '.dat')

t=np.arange(1, 51)/50

# Create 4 subplots sharing the same x-axis
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

# 1, mean density
axs[0].plot(cdl_5[:,1],cdl_5[:,0],label=r'$\mathcal{M}_0=5$')
axs[0].plot(cdl_10[:,1],cdl_10[:,0],label=r'$\mathcal{M}_0=10$')
axs[0].plot(cdl_15[:,1],cdl_15[:,0],label=r'$\mathcal{M}_0=15$')
axs[0].plot(cdl_20[:,1],cdl_20[:,0],label=r'$\mathcal{M}_0=20$')
axs[0].set_ylabel(r'$\bar{\rho}_\mathrm{cdl}$')
axs[0].set_yscale('log')
axs[0].legend()

# 2, mean rms Mach number
axs[1].plot(cdl_5[:,1],cdl_5[:,3],label=r'$\mathcal{M}_0=5$')
# axs[1].plot(cdl_10[:,1],cdl_10[:,3],label=r'$\mathcal{M}_0=10$')
# axs[1].plot(cdl_15[:,1],cdl_15[:,3],label=r'$\mathcal{M}_0=15$')
# axs[1].plot(cdl_20[:,1],cdl_20[:,3],label=r'$\mathcal{M}_0=20$')
axs[1].plot(cdl_10[:,1],cdl_10[:,16],'--',label=r'$\mathcal{M}_0=10$')
axs[1].plot(cdl_15[:,1],cdl_15[:,16],'--',label=r'$\mathcal{M}_0=15$')
axs[1].plot(cdl_20[:,1],cdl_20[:,16],'--',label=r'$\mathcal{M}_0=20$')
axs[1].set_ylabel(r'$\mathcal{M}_\mathrm{rms,\,cdl}$')
axs[1].set_yscale('log')

# 3, mean rms magnetic field
axs[2].plot(cdl_5[:,1],cdl_5[:,5],label=r'$\mathcal{M}_0=5$')
axs[2].plot(cdl_10[:,1],cdl_10[:,5],label=r'$\mathcal{M}_0=10$')
axs[2].plot(cdl_15[:,1],cdl_15[:,5],label=r'$\mathcal{M}_0=15$')
axs[2].plot(cdl_20[:,1],cdl_20[:,5],label=r'$\mathcal{M}_0=20$')
axs[2].set_ylabel(r'$B_\mathrm{rms,\,cdl}$')
axs[2].set_yscale('log')

# 4, normalized time
axs[3].plot(cdl_5[:,1],t*4*5,label=r'$\mathcal{M}_0=5$')
axs[3].plot(cdl_10[:,1],t*2*10,label=r'$\mathcal{M}_0=10$')
axs[3].plot(cdl_15[:,1],t*1*15,label=r'$\mathcal{M}_0=15$')
axs[3].plot(cdl_20[:,1],t*1*20,label=r'$\mathcal{M}_0=20$')
axs[3].set_ylabel(r'$\mathcal{M}_0 t$')
axs[3].set_xlabel("l")

plt.subplots_adjust(hspace=0)


out_dir = 'figs/' + run_dir
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + '/val_t_wei.pdf', format='pdf', bbox_inches='tight')