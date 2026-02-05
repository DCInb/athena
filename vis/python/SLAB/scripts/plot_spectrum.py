import sys
sys.path.append("../../")
import os
import numpy as np
from athena_read import athdf
import matplotlib.pyplot as plt
import Constants 
c=Constants.Constants()

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

def get_fft_var(d, var_name, id_xl, id_xr):
    # Hanning window
    shape_x = id_xr - id_xl
    wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(shape_x) / (shape_x - 1)))
    W  = wx[None, None, :]

    # Perform FFT on the 3D array
    if var_name == 'Bcc':
        fft_var1 = np.fft.fftshift(np.fft.fftn(d['Bcc1'][:,:,id_xl:id_xr]*W))
        fft_var2 = np.fft.fftshift(np.fft.fftn(d['Bcc2'][:,:,id_xl:id_xr]*W))
        fft_var3 = np.fft.fftshift(np.fft.fftn(d['Bcc3'][:,:,id_xl:id_xr]*W))
        fft_var = (fft_var1, fft_var2, fft_var3)
    elif var_name == 'va':
        fft_var1 = np.fft.fftshift(np.fft.fftn(d['Bcc1'][:,:,id_xl:id_xr]/d['rho'][:,:,id_xl:id_xr]*W))
        fft_var2 = np.fft.fftshift(np.fft.fftn(d['Bcc2'][:,:,id_xl:id_xr]/d['rho'][:,:,id_xl:id_xr]*W))
        fft_var3 = np.fft.fftshift(np.fft.fftn(d['Bcc3'][:,:,id_xl:id_xr]/d['rho'][:,:,id_xl:id_xr]*W))
    else:
        raise ValueError("Unsupported variable name for FFT.")

    return fft_var

def plot_powerlaw(ks, power):
    # Plot a reference power-law line for comparison
    k_ref = np.array([ks[1], ks[-1]])  # Avoid the zero frequency bin
    E_ref = 0.1*(k_ref/10)**power
    plt.plot(k_ref, E_ref, 'k--', label=f'k^{power}')

def plot_vark(myfile, var_name='Bcc', ax=None, plot_ref=False):
    '''
    Plot the spectrum of a given variable from an Athena++ output file.
    '''
    d = athdf(myfile)
    X,Y = np.meshgrid(d['x1f'], d['x2f'], indexing='ij')
    # Perform FFT on the 3D array
    shape = np.array(np.shape((d['rho'])))
    N=shape[0]
    dx=(d['x3f'][-1]-d['x3f'][0])/N
    dv=dx**3
    
    # Define the slice range in the x-
    shape_x=shape[2]
    id_xl = int(shape[2]/2 - shape_x/2)
    id_xr = int(shape[2]/2 + shape_x/2)

    # Generate wavevector coordinates ranging from -k to k
    kz = np.fft.fftfreq(shape[0],d=dx)
    ky = np.fft.fftfreq(shape[1],d=dx)
    kx = np.fft.fftfreq(shape_x, d=dx)

    # Shift zero frequency component to the center
    kz = np.fft.fftshift(kz)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)

    # Create meshgrida for the wavevectors
    Kz_3d, Ky_3d, Kx_3d = np.meshgrid(kz, ky, kx, indexing='ij')
    Kz_2d, Ky_2d = np.meshgrid(kz, ky, indexing='ij')
    kmag = np.sqrt(Kx_3d**2 + Ky_3d**2 + Kz_3d**2)

    # Get FFT of the field
    fft_var = get_fft_var(d, var_name, id_xl, id_xr)
    n_cells = shape[0]*shape[1]*shape_x
    power_spectrum_pa = np.abs(fft_var[0])**2/n_cells*dv
    power_spectrum_pe = (np.abs(fft_var[1])**2+np.abs(fft_var[2])**2)/n_cells*dv/2

    # Bin the k-values to compute the spectrum
    k_bins = np.linspace(np.min(kmag), np.max(kmag), num=500)
    energy_spectrum = np.zeros(len(k_bins) - 1)
    energy_spectrum_pa = np.zeros(len(k_bins) - 1)
    energy_spectrum_pe = np.zeros(len(k_bins) - 1)

    # Compute the energy in each bin
    for i in range(len(k_bins) - 1):
        bin_mask = (kmag >= k_bins[i]) & (kmag < k_bins[i + 1])
        dk = k_bins[i + 1] - k_bins[i]
        # energy_spectrum[i] = np.sum(power_spectrum[bin_mask])
        energy_spectrum_pa[i] = np.sum(power_spectrum_pa[bin_mask])/dk
        energy_spectrum_pe[i] = np.sum(power_spectrum_pe[bin_mask])/dk

    # Compute the center of each bin for plotting
    bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Plot the distribution
    num = myfile.split('.')[-2]   # '00030'
    n = int(num)
    ax[0].plot(bin_centers[1:], energy_spectrum_pe[1:],label=str(n))
    ax[1].plot(bin_centers[1:], energy_spectrum_pa[1:])

    if plot_ref:
        plot_powerlaw(bin_centers[1:], power=3/2)

def plot_vark_t(base_dir, var_name='Bcc'):
    fig, ax = plt.subplots(1, 2, figsize=(10,4))  # 1 row, 2 columns

    file_interval = 5
    files=[base_dir + "/COLL.out1."+"{0:05d}".format(i)+".athdf" for i in range(10,51,file_interval)]
    for i, myfile in enumerate(files):
        plot_vark(myfile, var_name, ax, plot_ref=(i==0))

    ax[0].set_xlabel(r'$|\mathbf{k}|L_y/2\pi$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylim(10**-3,10**0)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlabel(r'$|\mathbf{k}|L_y/2\pi$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(10**-3,10**0)
    ax[1].grid(True)

    if var_name == 'Bcc':
        ax[0].set_ylabel(r'$\mathcal{E}_{B\perp}(k)$')
        ax[1].set_ylabel(r'$\mathcal{E}_{B\parallel}(k)$')
    elif var_name == 'va':
        ax[0].set_ylabel(r'$\mathcal{E}_{v\perp}(k)$')
        ax[1].set_ylabel(r'$\mathcal{E}_{v\parallel}(k)$')
    else:
        raise ValueError("Unsupported variable name for labeling.")


    out_dir = '../figs/' + run_dir
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + '/EB_spectrum_t.pdf', format='pdf', bbox_inches='tight')

base_dir = '../../../../data/TDSC/'
run_dir = 'M20_B0.1_R1_D0.02_PR/'
base_dir += run_dir
plot_vark_t(base_dir, var_name='Bcc')