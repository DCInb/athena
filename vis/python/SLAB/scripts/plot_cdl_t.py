import numpy as np
import matplotlib.pyplot as plt

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

cdl_1 = np.loadtxt('../dat/M20_B0.1_R2_D0.02_PR.dat')
cdl_01 = np.loadtxt('../dat/M20_B0.01_R2_D0.02_PR.dat')
cdl_001 = np.loadtxt('../dat/M20_B0.001_R2_D0.02_PR.dat')
cdl_5 = np.loadtxt('../dat/M20_B0.5_R2_D0.02_PR.dat')
cdl_pe = np.loadtxt('../dat/M20_Bpe_R2_D0.02_PR.dat')

plt.plot(cdl_01[:,1],cdl_001[:,3],'#87e000')
plt.plot(cdl_01[:,1],cdl_01[:,3],'#87e885')
plt.plot(cdl_1[:,1],cdl_1[:,3],'#fae768')
plt.plot(cdl_5[:,1],cdl_5[:,3],'#fa8080')
plt.plot(cdl_pe[:,1],cdl_pe[:,3],'k')
plt.plot(cdl_01[:,1],cdl_001[:,4],linestyle='--', color='#87e000')
plt.plot(cdl_01[:,1],cdl_01[:,4],linestyle='--', color='#87e885')
plt.plot(cdl_1[:,1],cdl_1[:,4],linestyle='--', color='#fae768')
plt.plot(cdl_5[:,1],cdl_5[:,4],linestyle='--', color='#fa8080')
plt.plot(cdl_pe[:,1],cdl_pe[:,4],linestyle='--', color='k')

# plt.ylim(10**-3,10**0.5)
plt.axvline(x=-1)
plt.xlim(0,20)
plt.yscale('log')
plt.xlabel('l')
plt.ylabel(r'$Mach_{cdl}$')
plt.savefig('../figs/mach_cdl.pdf', format='pdf', bbox_inches='tight')