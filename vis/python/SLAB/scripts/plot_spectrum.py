import sys
sys.path.append("../../")
import os
import numpy as np
from athena_read import athdf
import matplotlib.pyplot as plt
import Constants 
c=Constants.Constants()

base_dir = '../../../../data/TDSC/'
run_dir = 'M20_B0.5_R2_D0.02_PR/'
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

def hemholtz_decomposition(d):

    # Perform FFT on the 3D array
    shape = np.array(np.shape((d['rho'])))

    # Generate wavevector coordinates ranging from -k to k
    kz = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kx = np.fft.fftfreq(shape[2])

    # Shift zero frequency component to the center
    kz = np.fft.fftshift(kz)*(shape[0]/1)
    ky = np.fft.fftshift(ky)*(shape[1]/1)
    kx = np.fft.fftshift(kx)*(shape[2]/2)

    # Create meshgrida for the wavevectors
    Kz_3d, Ky_3d, Kx_3d = np.meshgrid(kz, ky, kx, indexing='ij')
    Kz_2d, Ky_2d = np.meshgrid(kz, ky, indexing='ij')
    k2_3d = (Kx_3d**2 + Ky_3d**2 + Kz_3d**2)
    k2_2d = Ky_2d**2 + Kz_2d**2

    fft_vx = np.fft.fftshift(np.fft.fftn(d['vel1'])) #, axes=(0, 1)
    fft_vy = np.fft.fftshift(np.fft.fftn(d['vel2']))
    fft_vz = np.fft.fftshift(np.fft.fftn(d['vel3']))

    fft_vx_c = (Kx_3d*fft_vx+Ky_3d*fft_vy+Kz_3d*fft_vz)*Kx/(k2_3d+1e-20)
    fft_vy_c = (Kx_3d*fft_vx+Ky_3d*fft_vy+Kz_3d*fft_vz)*Ky/(k2_3d+1e-20)
    fft_vz_c = (Kx_3d*fft_vx+Ky_3d*fft_vy+Kz_3d*fft_vz)*Kz/(k2_3d+1e-20)

    fft_vx_s = fft_vx-fft_vx_c
    fft_vy_s = fft_vy-fft_vy_c
    fft_vz_s = fft_vz-fft_vz_c

    vx_s = np.fft.ifftn(np.fft.ifftshift(fft_vx_s))
    vy_s = np.fft.ifftn(np.fft.ifftshift(fft_vy_s))
    vz_s = np.fft.ifftn(np.fft.ifftshift(fft_vz_s))

    vx_c = np.fft.ifftn(np.fft.ifftshift(fft_vx_c))
    vy_c = np.fft.ifftn(np.fft.ifftshift(fft_vy_c))
    vz_c = np.fft.ifftn(np.fft.ifftshift(fft_vz_c))

    return vx_s,vy_s,vz_s, vx_c,vy_c,vz_c


def plot_Bk_t(file):

    d = athdf(myfile)
    X,Y = np.meshgrid(d['x1f'], d['x2f'], indexing='ij')
    # Perform FFT on the 3D array
    shape = np.array(np.shape((d['rho'])))
    shape_x = shape[2]
    shape_x = 60

    # Generate wavevector coordinates ranging from -k to k
    kz = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kx = np.fft.fftfreq(shape_x)

    # Shift zero frequency component to the center
    kz = np.fft.fftshift(kz)*(shape[0]/1)
    ky = np.fft.fftshift(ky)*(shape[1]/1)
    kx = np.fft.fftshift(kx)*(shape[2]/2)

    # Create meshgrida for the wavevectors
    Kz_3d, Ky_3d, Kx_3d = np.meshgrid(kz, ky, kx, indexing='ij')
    Kz_2d, Ky_2d = np.meshgrid(kz, ky, indexing='ij')

    # Hanning window
    wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(shape_x) / (shape_x - 1)))
    W  = wx[None, None, :]

    # id_xl=0
    # id_xr=-1
    id_xl=226
    id_xr=286

    vx_s,vy_s,vz_s, vx_c,vy_c,vz_c = hemholtz_decomposition(d)


    fft_vx = np.fft.fftshift(np.fft.fftn(d['vel1'][:,:,id_xl:id_xr]*W)) #, axes=(0, 1)
    fft_vy = np.fft.fftshift(np.fft.fftn(d['vel2'][:,:,id_xl:id_xr]*W))
    fft_vz = np.fft.fftshift(np.fft.fftn(d['vel3'][:,:,id_xl:id_xr]*W))

    fft_vx_s = np.fft.fftshift(np.fft.fftn(vx_s[:,:,id_xl:id_xr]*W))
    fft_vy_s = np.fft.fftshift(np.fft.fftn(vy_s[:,:,id_xl:id_xr]*W))
    fft_vz_s = np.fft.fftshift(np.fft.fftn(vz_s[:,:,id_xl:id_xr]*W))

    fft_vx_c = np.fft.fftshift(np.fft.fftn(vx_c[:,:,id_xl:id_xr]*W))
    fft_vy_c = np.fft.fftshift(np.fft.fftn(vx_c[:,:,id_xl:id_xr]*W))
    fft_vz_c = np.fft.fftshift(np.fft.fftn(vx_c[:,:,id_xl:id_xr]*W))


    fft_Bx = np.fft.fftshift(np.fft.fftn((d['Bcc1'][:,:,id_xl:id_xr])*W))
    fft_By = np.fft.fftshift(np.fft.fftn(d['Bcc2'][:,:,id_xl:id_xr]*W))
    fft_Bz = np.fft.fftshift(np.fft.fftn(d['Bcc3'][:,:,id_xl:id_xr]*W))

    fft_vax = np.fft.fftshift(np.fft.fftn((d['Bcc1'][:,:,id_xl:id_xr])/d['rho'][:,:,id_xl:id_xr]))
    fft_vay = np.fft.fftshift(np.fft.fftn(d['Bcc2'][:,:,id_xl:id_xr]/d['rho'][:,:,id_xl:id_xr]))
    fft_vaz = np.fft.fftshift(np.fft.fftn(d['Bcc3'][:,:,id_xl:id_xr]/d['rho'][:,:,id_xl:id_xr]))

    k = np.sqrt(Kx_3d**2 + Ky_3d**2 + Kz_3d**2)
    power_spectrum = np.abs(fft_vx)**2+np.abs(fft_vy)**2+np.abs(fft_vz)**2
    power_spectrum_vpa = np.abs(fft_vx_s)**2
    power_spectrum_vpe = np.abs(fft_vy_s)**2+np.abs(fft_vz_s)**2

    # Bin the k-values to compute the spectrum
    k_bins = np.linspace(np.min(k), (np.max(k)), num=500)  # Adjust bin size as needed
    energy_spectrum = np.zeros(len(k_bins) - 1)
    energy_spectrum_vpa = np.zeros(len(k_bins) - 1)
    energy_spectrum_vpe = np.zeros(len(k_bins) - 1)

    # Compute the energy in each bin
    for i in range(len(k_bins) - 1):
        bin_mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
        energy_spectrum[i] = np.sum(power_spectrum[bin_mask])
        energy_spectrum_vpa[i] = np.sum(power_spectrum_vpa[bin_mask])
        energy_spectrum_vpe[i] = np.sum(power_spectrum_vpe[bin_mask])

    # Compute the center of each bin for plotting
    bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Plot the distribution
    num = myfile.split('.')[-2]   # '00030'
    n = int(num)
    ax[0].plot(bin_centers[1:], energy_spectrum_vpe[1:],label=str(n))
    ax[1].plot(bin_centers[1:], energy_spectrum_vpa[1:])
    # plt.plot(bin_centers[1:], energy_spectrum_vpe[1:], label=r'$\mathcal{E}_{\perp}(k)$')
    # plt.plot(bin_centers[1:], energy_spectrum_vpa[1:], label=r'$\mathcal{E}_{\parallel}(k)$')
    # plt.plot(bin_centers[100:250],bin_centers[100:250]**(-1)*10**9,'k--',label='slope=-1')
    # plt.plot(bin_centers[100:200],bin_centers[100:200]**(-1)*10**8.5,'k--')


fig, ax = plt.subplots(1, 2, figsize=(10,4))  # 1 row, 2 columns

myfile=base_dir + "COLL.out1."+"00010"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00015"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00020"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00025"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00030"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00035"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00040"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00045"+".athdf"
plot_Bk_t(myfile)
myfile=base_dir + "COLL.out1."+"00050"+".athdf"
plot_Bk_t(myfile)

ax[0].set_xlabel(r'$|k|$')
ax[0].set_ylabel(r'$\mathcal{E}_{v_s\perp}(k)$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim(10**6,10**9)
ax[0].grid(True)
ax[0].legend()

ax[1].set_xlabel(r'$|k|$')
ax[1].set_ylabel(r'$\mathcal{E}_{v_s\parallel}(k)$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_ylim(10**7,10**10)
ax[1].grid(True)


out_dir = '../figs/' + run_dir
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_dir + '/Ek_s_spectrum_t.pdf', format='pdf', bbox_inches='tight')