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

run_dir = 'M10_B0.1_R1_D0.02_PR'
cdl_1 = np.loadtxt('../dat/' + run_dir + '.dat')
run_dir = 'M10_B0.1_R4_D0.02_PR'
cdl_4 = np.loadtxt('../dat/' + run_dir + '.dat')
run_dir = 'M10_B0.1_R2_D0.02_PR'
cdl_2 = np.loadtxt('../dat/' + run_dir + '.dat')
t=np.arange(1, 51)/50

plt.plot(t,cdl_1[:,5],'r',label=r'$M_{cdl}$')
plt.plot(t,cdl_1[:,6],'r--',label=r'$M_{\perp}$')
plt.plot(t,cdl_1[:,7],'r-.',label=r'$M_{\parallel}$')
plt.plot(t,cdl_2[:,5],'b')
plt.plot(t,cdl_2[:,6],'b--')
plt.plot(t,cdl_2[:,7],'b-.')
plt.plot(t,cdl_4[:,5],'g')
plt.plot(t,cdl_4[:,6],'g--')
plt.plot(t,cdl_4[:,7],'g-.')


# Plot a auxiliary line with slope of growth rate
plt.plot(t[4:10], 0.01*np.exp(30*(t[4:10]-0.1)), 'k--', label=r'$/gamma=30$')    

plt.axvline(x=-1)
plt.xlim(0,1)
# plt.ylim(10**-2,10**-0.3)
plt.yscale('log')
plt.xlabel('t')
plt.ylabel(r'$B_{\rm rms}$')
plt.legend()

out_dir = '../figs/' + run_dir
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + '/B_t_1D.pdf', format='pdf', bbox_inches='tight')