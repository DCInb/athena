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

def init_data(myfile):
    """
    Initialize data from an Athena++ HDF5 output file.
    """
    d = athdf(myfile)
    X,Y = np.meshgrid(d['x1f'], d['x2f'], indexing='ij')
    # Perform FFT on the 3D array
    shape = np.array(np.shape((d['rho'])))
    N=shape[0]
    shape_x = shape[2]

    # Generate wavevector coordinates ranging from -k to k
    # This is k/2\pi*L_y
    kz = np.fft.fftfreq(shape[0], d=2/N)*2
    ky = np.fft.fftfreq(shape[1], d=2/N)*2
    kx = np.fft.fftfreq(shape_x, d=2/N)*2

    # Shift zero frequency component to the center
    kz = np.fft.fftshift(kz)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)

    # Create meshgrida for the wavevectors
    Kz_3d, Ky_3d, Kx_3d = np.meshgrid(kz, ky, kx, indexing='ij')
    Kz_2d, Ky_2d = np.meshgrid(kz, ky, indexing='ij')
    Kmag = np.sqrt(Kx_3d**2 + Ky_3d**2 + Kz_3d**2)

    # FFT of the fields
    wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(shape_x) / (shape_x - 1)))
    W  = wx[None, None, :]

    fft_vx = np.fft.fftshift(np.fft.fftn(d['vel1']*np.sqrt(d['rho'])*W))
    fft_vy = np.fft.fftshift(np.fft.fftn(d['vel2']*np.sqrt(d['rho'])*W))
    fft_vz = np.fft.fftshift(np.fft.fftn(d['vel3']*np.sqrt(d['rho'])*W))
    
    fft_bx = np.fft.fftshift(np.fft.fftn(d['Bcc1']*W))
    fft_by = np.fft.fftshift(np.fft.fftn(d['Bcc2']*W))
    fft_bz = np.fft.fftshift(np.fft.fftn(d['Bcc3']*W))

    cs = 1
    fft_p = np.fft.fftshift(np.fft.fftn(d['rho']*cs**2*W))

    return d, (fft_vx, fft_vy, fft_vz), (fft_bx, fft_by, fft_bz), fft_p, Kmag, 2/N


def band_pass(d_fft, kl, kr, Kmag):
    """
    Apply a shell filter in Fourier space, return the filtered data in real space.
    """
    # Adjust odd
    if kr<kl:
        kr, kl = kl, kr
    # Apply filter
    mask = (Kmag >= kl) & (Kmag < kr)
    d_fft_filtered = d_fft * mask
    # Inverse FFT to get back to real space
    d_filtered = np.fft.ifftn(np.fft.ifftshift(d_fft_filtered))
    return d_filtered.real

def tau_uua(vK, vQ, d, dx):
    """
    Compute the energy transfer rate through advection from velocity field vQ to velocity field vK.
    """
    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        # Compute gradients of vQ
        dvQ_dz, dvQ_dy, dvQ_dx = np.gradient(vQ[i], dx, dx, dx)
        tau += - vK[i] * (d['vel1'] * dvQ_dx + d['vel2'] * dvQ_dy + d['vel3'] * dvQ_dz)
    return np.sum(tau)

def tau_uuc(vK, vQ, d, dx):
    """
    Compute the energy transfer rate through compression from velocity field vK to velocity field vQ.
    """
    # Compute the divergence of v
    dvx_dx = np.gradient(d['vel1'], dx, axis=2)
    dvy_dy = np.gradient(d['vel2'], dx, axis=1)
    dvz_dz = np.gradient(d['vel3'], dx, axis=0)
    div_v = dvx_dx + dvy_dy + dvz_dz

    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        tau += - vK[i] * vQ[i] * div_v
    return np.sum(tau)

def tau_bba(bK, bQ, d, dx):
    """
    Compute the energy transfer rate through advection from magnetic field bQ to magnetic field bK.
    """
    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        # Compute gradients of bQ
        dbQ_dz, dbQ_dy, dbQ_dx = np.gradient(bQ[i], dx, dx, dx)
        tau += - bK[i] * (d['vel1'] * dbQ_dx + d['vel2'] * dbQ_dy + d['vel3'] * dbQ_dz)
    return np.sum(tau)

def tau_bbc(bK, bQ, d, dx):
    """
    Compute the energy transfer rate through compression from magnetic field bK to magnetic field bQ.
    """
    # Compute the divergence of v
    dvx_dx = np.gradient(d['vel1'], dx, axis=2)
    dvy_dy = np.gradient(d['vel2'], dx, axis=1)
    dvz_dz = np.gradient(d['vel3'], dx, axis=0)
    div_v = dvx_dx + dvy_dy + dvz_dz

    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        tau += - bK[i] * bQ[i] * div_v
    return np.sum(tau)

def tau_bup(uK, bQ, d, dx):
    """
    Compute the energy transfer rate through magnetic pressure from magnetic field bQ to velocity field uK.
    """
    # Magnetic pressure term
    bb = (d['Bcc1']*bQ[0] + d['Bcc2']*bQ[1] + d['Bcc3']*bQ[2]) / 2 
    # Compute gradients of bb
    dbb_dz, dbb_dy, dbb_dx = np.gradient(bb, dx, dx, dx)

    # Compute the energy transfer rate
    tau = - (uK[0] * dbb_dx + uK[1] * dbb_dy + uK[2] * dbb_dz)/np.sqrt(d['rho'])
    return np.sum(tau)

def tau_but(uK, bQ, d, dx):
    """
    Compute the energy transfer rate through magnetic tension from magnetic field bQ to velocity field uK.
    """
    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        # Compute gradients of bQ
        dbQ_dz, dbQ_dy, dbQ_dx = np.gradient(bQ[i], dx, dx, dx)
        tau += uK[i] * (d['Bcc1']/np.sqrt(d['rho']) * dbQ_dx + d['Bcc2']/np.sqrt(d['rho']) * dbQ_dy + d['Bcc3']/np.sqrt(d['rho']) * dbQ_dz)
    return np.sum(tau)

def tau_ubp(bK, uQ, d, dx):
    """
    Compute the energy transfer rate through magnetic pressure from velocity field uQ to magnetic field bK.
    """
    # Compute divergence of uQ
    duQ_dz, duQ_dy, duQ_dx = np.gradient(uQ[0]/np.sqrt(d['rho']), dx, dx, dx)
    div_uQ = duQ_dx + duQ_dy + duQ_dz

    # Compute the energy transfer rate
    tau = - (bK[0] * d['Bcc1'] + bK[1] * d['Bcc2'] + bK[2] * d['Bcc3']) * div_uQ / 2
    return np.sum(tau)

def tau_ubt(bK, uQ, d, dx):
    """
    Compute the energy transfer rate through magnetic tension from magnetic field bK to velocity field uQ.
    """
    # Compute the energy transfer rate
    tau = np.zeros_like(d['rho'])
    for i in range(3):
        # Compute divergence of uQ
        duQ_dx = np.gradient(uQ[i] * d['Bcc1']/np.sqrt(d['rho']), dx, axis=2)
        duQ_dy = np.gradient(uQ[i] * d['Bcc2']/np.sqrt(d['rho']), dx, axis=1)
        duQ_dz = np.gradient(uQ[i] * d['Bcc3']/np.sqrt(d['rho']), dx, axis=0)
        div_uQ = duQ_dx + duQ_dy + duQ_dz
        tau += bK[i] * div_uQ
    return np.sum(tau)

def tau_pu(uK, pQ, d, dx):
    """
    Compute the energy transfer rate through pressure from pressure field pQ to velocity field uK.
    Becareful pQ is a scalar field.
    """
    # Compute gradients of pQ
    dpQ_dz, dpQ_dy, dpQ_dx = np.gradient(pQ, dx, dx, dx)

    # Compute the energy transfer rate
    tau = - (uK[0] * dpQ_dx + uK[1] * dpQ_dy + uK[2] * dpQ_dz)/np.sqrt(d['rho'])
    return np.sum(tau)

def cross_scale(k_mid, d, dx, fft_vx, fft_vy, fft_vz, fft_bx, fft_by, fft_bz, fft_p, Kmag):
    """
    Compute cross-scale energy transfer rates at a given mid wavenumber k_mid.
    """
    # Apply band-pass filters, from large scale to small scale
    vK = [band_pass(fft_vx, k_mid, np.inf, Kmag),
          band_pass(fft_vy, k_mid, np.inf, Kmag),
          band_pass(fft_vz, k_mid, np.inf, Kmag)]
    vQ = [band_pass(fft_vx, 0, k_mid, Kmag),
          band_pass(fft_vy, 0, k_mid, Kmag),
          band_pass(fft_vz, 0, k_mid, Kmag)]
    bK = [band_pass(fft_bx, k_mid, np.inf, Kmag),
          band_pass(fft_by, k_mid, np.inf, Kmag),
          band_pass(fft_bz, k_mid, np.inf, Kmag)]
    bQ = [band_pass(fft_bx, 0, k_mid, Kmag),
          band_pass(fft_by, 0, k_mid, Kmag),
          band_pass(fft_bz, 0, k_mid, Kmag)]
    pQ = band_pass(fft_p, 0, k_mid, Kmag)

    # Compute energy transfer rates
    tau_UUA = tau_uua(vK, vQ, d, dx)
    tau_UUC = tau_uuc(vK, vQ, d, dx)
    tau_BBA = tau_bba(bK, bQ, d, dx)
    tau_BBC = tau_bbc(bK, bQ, d, dx)
    tau_BUP = tau_bup(vK, bQ, d, dx)
    tau_BUT = tau_but(vK, bQ, d, dx)
    tau_UBP = tau_ubp(bK, vQ, d, dx)
    tau_UBT = tau_ubt(bK, vQ, d, dx)
    tau_PU  = tau_pu(vK, pQ, d, dx)

    tau_tot = (tau_UUA + tau_UUC + tau_BBA + tau_BBC + tau_BUP + tau_BUT + tau_UBP + tau_UBT + tau_PU)

    return (tau_tot, tau_UUA, tau_UUC, tau_BUT, tau_BUP, tau_UBT, tau_UBP, tau_BBA, tau_BBC, tau_PU)

def plot_cro(myfile, n_plot=10):
    """
    Plot cross-scale energy transfer rates.
    """
    # Initialize data
    d, (fft_vx, fft_vy, fft_vz), (fft_bx, fft_by, fft_bz), fft_p, Kmag, dx = init_data(myfile)
    k_mids = np.logspace(0, np.log10(np.max(Kmag)), num=n_plot)
    taus =  [[] for _ in range(10)]
    for i in range(n_plot):
        tau = cross_scale(k_mids[i], d, dx, fft_vx, fft_vy, fft_vz, fft_bx, fft_by, fft_bz, fft_p, Kmag)
        for j in range(10):
            taus[j].append(tau[j])
    # Convert to numpy arrays
    taus = np.array(taus)
    
    # Plot the results
    names = [
        r'$\mathcal{T}_{\mathrm{tot}}$',
        r'$\mathcal{T}_{\mathrm{UUa}}$',
        r'$\mathcal{T}_{\mathrm{UUc}}$',
        r'$\mathcal{T}_{\mathrm{BUT}}$',
        r'$\mathcal{T}_{\mathrm{BUP}}$',
        r'$\mathcal{T}_{\mathrm{UBT}}$',
        r'$\mathcal{T}_{\mathrm{UBP}}$',
        r'$\mathcal{T}_{\mathrm{BBa}}$',
        r'$\mathcal{T}_{\mathrm{BBc}}$',
        r'$\mathcal{T}_{\mathrm{PU}}$'
    ]

    colors = [
        '#1b9e77', 
        '#f2a7a1', '#d7301f',
        '#b8e186', '#66a61e',
        '#bcd7e8', '#4575b4',
        '#fdd08d', '#f68026',
        '#d95f02'
    ]

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(6, 4))
    axes = axes.ravel()

    # Fig.0, total
    axes[0].plot(k_mids, taus[0], color=colors[0], label=names[0])
    axes[0].plot(k_mids, taus[1]+taus[2], '--', color=colors[1], label=r'$\mathcal{T}_{\mathrm{UU}}$')
    axes[0].plot(k_mids, taus[3]+taus[4], '--', color=colors[3], label=r'$\mathcal{T}_{\mathrm{BU}}$')
    axes[0].plot(k_mids, taus[5]+taus[6], '--', color=colors[5], label=r'$\mathcal{T}_{\mathrm{UB}}$')
    axes[0].plot(k_mids, taus[7]+taus[8], '--', color=colors[7], label=r'$\mathcal{T}_{\mathrm{BB}}$')
    axes[0].plot(k_mids, taus[9], color=colors[9], label=r'$\mathcal{T}_{\mathrm{PU}}$')

    axes[0].set_xscale('log')
    axes[0].legend()
    
    # Fig.1, PU
    axes[1].plot(k_mids, taus[9], color=colors[9], label=r'$\mathcal{T}_{\mathrm{PU}}$')
    axes[1].set_xscale('log')

    # Fig.2, UU
    axes[2].plot(k_mids, taus[1], color=colors[1], label=names[1])
    axes[2].plot(k_mids, taus[2], color=colors[2], label=names[2])
    axes[2].plot(k_mids, taus[1]+taus[2], '--', color=colors[1])
    axes[2].set_xscale('log')
    axes[2].legend()

    # Fig.3, BB
    axes[3].plot(k_mids, taus[7], color=colors[7], label=names[7])
    axes[3].plot(k_mids, taus[8], color=colors[8], label=names[8])
    axes[3].plot(k_mids, taus[7]+taus[8], '--', color=colors[7])
    axes[3].set_xscale('log')
    axes[3].legend()

    # Fig.4, BU
    axes[4].plot(k_mids, taus[3], color=colors[3], label=names[3])
    axes[4].plot(k_mids, taus[4], color=colors[4], label=names[4])
    axes[4].plot(k_mids, taus[3]+taus[4], '--', color=colors[3])
    axes[4].set_xscale('log')
    axes[4].legend()

    # Fig.5, UB
    axes[5].plot(k_mids, taus[5], color=colors[5], label=names[5])
    axes[5].plot(k_mids, taus[6], color=colors[6], label=names[6])
    axes[5].plot(k_mids, taus[5]+taus[6], '--', color=colors[5])
    axes[5].set_xscale('log')
    axes[5].legend()

    # Add labels
    axes[4].set_xlabel(r'$k_{\mathrm{mid}}L_y/2\pi$')
    axes[5].set_xlabel(r'$k_{\mathrm{mid}}L_y/2\pi$')
    axes[0].set_ylabel(r'$\mathcal{T}_{\mathrm{XY}}^{k_\mathrm{mid}}$')
    axes[2].set_ylabel(r'$\mathcal{T}_{\mathrm{XY}}^{k_\mathrm{mid}}$')
    axes[4].set_ylabel(r'$\mathcal{T}_{\mathrm{XY}}^{k_\mathrm{mid}}$')
    plt.tight_layout()

    # Save the figure
    out_dir = '../figs/' + run_dir
    os.makedirs(out_dir, exist_ok=True)
    num = myfile.split('.')[-2]   # '00030'
    n = str(int(num))
    plt.savefig(out_dir + '/cro_t' + n + '.pdf', format='pdf', bbox_inches='tight')

base_dir = '../../../../data/TDSC/'
run_dir = 'M10_B0.1_R2_D0.02_PR/'
base_dir += run_dir
myfile=base_dir + "COLL.out1."+"00010"+".athdf"
plot_cro(myfile)