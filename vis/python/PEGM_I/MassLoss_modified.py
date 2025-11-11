import numpy as np
import Constants 
from athena_read import athdf
import scipy

c=Constants.Constants()

# plot unbounded mass by Bernoulli constant
def find_r(file, r_given):
    r_given_filelike = r_given*np.ones_like(file)
    r_index = np.argmin(np.abs(file-r_given_filelike))
    return r_index

def m_dot_parallel(i,myfile,phi_interp,gamma,rclist,m_dot_time,time,unbound=False,outflow=True):
    d = athdf(myfile)
    print(myfile+" file readed...")
    time[i] = d["Time"]
    
    v_abs = np.sqrt(d['vel1']**2+d['vel2']**2+d['vel3']**2)
    phi_r = phi_interp(d['x1v'])
    B = d['rho']*d['r0']*v_abs**2/2+d['rho']*d['r0']*phi_r+d['press']*(1+1/(gamma-1))
    
    ### caluculate the M_dot at rc
    for j,rc in enumerate(rclist):
        rc_index = find_r(d['x1v'],rc)
        d_theta = np.pi/d['rho'].shape[1]
        d_phi = 2*np.pi/d['rho'].shape[0]
        M_dot_tp = (d['rho']*d['r0']*d['vel1'])[:,:,rc_index]
        if outflow:
            M_dot_tp = np.where(d['vel1'][:,:,rc_index]>0,M_dot_tp,0)
        if unbound:
            M_dot_tp = np.where(B[:,:,rc_index]>=0,M_dot_tp,0)
        M_dot_tp = np.dot(M_dot_tp,np.sin(d['x2v']))
        M_dot = np.sum(M_dot_tp)*rc**2*d_theta*d_phi
        
        m_dot_time_i = m_dot_time[i]
        m_dot_time_i[j] = M_dot
        m_dot_time[i] = m_dot_time_i
        

def mass_loss(plot_num, plot_interv, rc_start, rc_num, rc_final, r_star,gamma, base_dir, unbound=False,outflow=True):
    """
        Use multiprocessing.Process to parallel execution
    """
    filelist = []

    for id_name in range(plot_num):
        datafile = base_dir + "PEGM.out1."+str(id_name*plot_interv).zfill(5)+".athdf"
        filelist.append(datafile)
        
    rclist = r_star*np.linspace(rc_start,rc_final,num=rc_num)
    
    r_phi = np.loadtxt(base_dir+"potential.dat")
    phi_interp = scipy.interpolate.interp1d(r_phi[:,0],r_phi[:,1])
    
    from multiprocessing import Pool,Array,Process,Manager
    
    p_list = []
    manager = Manager()
    m_dot_time = manager.list(np.zeros((plot_num,rc_num),dtype=np.float64))
    time = manager.list(np.zeros((plot_num,),dtype=np.float64))
     
    pool = Pool(20)
    for i,myfile in enumerate(filelist):
        pool.apply_async(m_dot_parallel, args=(i,myfile,phi_interp,gamma,rclist,m_dot_time,time,unbound,outflow))
    
    pool.close()
    pool.join()
    
    print("done!")
    
    return time,m_dot_time