import numpy as np
from athena_read import athdf


def get_xbound(d,target=0.8*20):
    d_shape = np.shape(d['vel1'])

    mask = np.flip(d['vel1'][:,:,d_shape[2]//2:],axis=2) > -target
    idx_r = np.argmax(mask, axis=2) 
    mask = d['vel1'][:,:,:d_shape[2]//2] < target
    idx_l = np.argmax(mask, axis=2) 
    x_r=np.flip(d['x1f'][d_shape[2]//2:])[idx_r]
    x_l=d['x1f'][idx_l]
    return x_l, x_r, idx_l, d_shape[2]-1-idx_r
    
def get_cdl(d, idx_l, idx_r):
    m_cdl  = 0     # mass
    n_cdl  = 0     # volume
    M2_tot = 0     # Mach number
    M2_perp= 0     # perpendicular component of Mach number
    M2_para= 0     # parallel component of Mach number
    B2_tot = 0     # mangetic field
    B2_perp= 0     # perpendicular component of mangetic field
    B2_para= 0     # parallel component of mangetic field
    va2_tot =0     # Alfven speed
    va2_perp=0     # perpendicular component of Alfven speed
    va2_para=0     # parallel component of Alfven speed
    b_abs   =0     # total magnetic field
    rho2    =0     # density squared
    cov     =0     # covariance of magnetic field and density field

    d_shape=np.shape(d['rho'])
    for k in range(d_shape[0]):
        for j in range(d_shape[1]):
            for i in range(idx_l[k,j], idx_r[k,j]+1):
                m_cdl   +=d['rho'][k,j,i]
                n_cdl   +=1
                M2_tot  +=d['vel1'][k,j,i]**2+d['vel2'][k,j,i]**2+d['vel3'][k,j,i]**2
                M2_perp +=d['vel2'][k,j,i]**2+d['vel3'][k,j,i]**2
                M2_para +=d['vel1'][k,j,i]**2
                B2_tot  +=d['Bcc1'][k,j,i]**2+d['Bcc2'][k,j,i]**2+d['Bcc3'][k,j,i]**2
                B2_perp +=d['Bcc2'][k,j,i]**2+d['Bcc3'][k,j,i]**2
                B2_para +=d['Bcc1'][k,j,i]**2
                va2_tot +=(d['Bcc1'][k,j,i]**2+d['Bcc2'][k,j,i]**2+d['Bcc3'][k,j,i]**2)/d['rho'][k,j,i]
                va2_perp+=(d['Bcc2'][k,j,i]**2+d['Bcc3'][k,j,i]**2)/d['rho'][k,j,i]
                va2_para+=(d['Bcc1'][k,j,i]**2)/d['rho'][k,j,i]
                B_ijk    =(d['Bcc1'][k,j,i]**2+d['Bcc2'][k,j,i]**2+d['Bcc3'][k,j,i]**2)**0.5
                b_abs   += B_ijk
                cov     += B_ijk*d['rho'][k,j,i]
                rho2    += d['rho'][k,j,i]**2

    rho_m   = m_cdl/n_cdl
    l_cdl   = n_cdl * (d['x1f'][-1]-d['x1f'][0])/d_shape[0]/d_shape[1]/d_shape[2]
    l       = m_cdl/d_shape[0]**3
    M_tot   = (M2_tot/n_cdl)**0.5
    M_perp  = (M2_perp/n_cdl)**0.5
    M_para  = (M2_para/n_cdl)**0.5
    B_tot   = (B2_tot/n_cdl)**0.5
    B_perp  = (B2_perp/n_cdl)**0.5
    B_para  = (B2_para/n_cdl)**0.5
    va_tot  = (va2_tot/n_cdl)**0.5
    va_perp = (va2_perp/n_cdl)**0.5
    va_para = (va2_para/n_cdl)**0.5
    cov     = (cov-b_abs*m_cdl/n_cdl)/((rho2-rho_m**2/n_cdl)*(B2_tot-b_abs**2/n_cdl))**0.5

    return rho_m, l_cdl, l, M_tot, M_perp, M_para, B_tot, B_perp, B_para, va_tot, va_perp, va_para, cov

def get_cdl_t(files,target=0.8*20):
    rho_m_t = np.zeros_like(files,dtype=np.float64)
    l_t     = np.zeros_like(files,dtype=np.float64)
    M_tot_t = np.zeros_like(files,dtype=np.float64)
    M_perp_t = np.zeros_like(files,dtype=np.float64)
    M_para_t = np.zeros_like(files,dtype=np.float64)
    B_tot_t = np.zeros_like(files,dtype=np.float64)
    B_perp_t = np.zeros_like(files,dtype=np.float64)
    B_para_t = np.zeros_like(files,dtype=np.float64)
    va_tot_t = np.zeros_like(files,dtype=np.float64)
    va_perp_t = np.zeros_like(files,dtype=np.float64)
    va_para_t = np.zeros_like(files,dtype=np.float64)
    rho_max_t = np.zeros_like(files,dtype=np.float64)
    M_tot_max_t = np.zeros_like(files,dtype=np.float64)
    B_tot_max_t = np.zeros_like(files,dtype=np.float64)
    va_tot_max_t = np.zeros_like(files,dtype=np.float64)
    cov_t = np.zeros_like(files,dtype=np.float64)

    for id, myfile in enumerate(files):
        d = athdf(myfile)
        x_l, x_r, idx_l, idx_r = get_xbound(d,target=target)
        rho_m, l_cdl, l, M_tot, M_perp, M_para, B_tot, B_perp, B_para, va_tot, va_perp, va_para, cov = get_cdl(d, idx_l, idx_r)
        rho_m_t[id]   = rho_m
        l_t[id]       = l
        M_tot_t[id]   = M_tot
        M_perp_t[id]  = M_perp
        M_para_t[id]  = M_para
        B_tot_t[id]   = B_tot
        B_perp_t[id]  = B_perp
        B_para_t[id]  = B_para
        va_tot_t[id]  = va_tot
        va_perp_t[id] = va_perp
        va_para_t[id] = va_para
        cov_t[id]     = cov
        
        rho_max_t[id]     = np.max(d['rho'])
        M_tot_max_t[id]   = np.max(d['vel1']**2+d['vel2']**2+d['vel3']**2)**0.5
        B_tot_max_t[id]   = np.max(d['Bcc1']**2+d['Bcc2']**2+d['Bcc3']**2)**0.5
        va_tot_max_t[id]  = np.max((d['Bcc1']**2+d['Bcc2']**2+d['Bcc3']**2)/d['rho'])**0.5
    
    return rho_m_t, l_t, M_tot_t, M_perp_t, M_para_t, B_tot_t, B_perp_t, B_para_t, va_tot_t, va_perp_t, va_para_t, rho_max_t, M_tot_max_t, B_tot_max_t, va_tot_max_t, cov_t

