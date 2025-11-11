import sys
sys.path.append("../")
import cdl
import numpy as np

base_dir = '../../../data/collision_denturb_M20/'
files = [base_dir + "COLL.out1."+ str(i).zfill(5) +".athdf" for i in range(1,21)]
rho_m_t, l_t, M_tot_t, M_perp_t, M_para_t, B_tot_t, B_perp_t, B_para_t, va_tot_t, va_perp_t, va_para_t, rho_max_t, M_tot_max_t, B_tot_max_t, va_tot_max_t = cdl.get_cdl_t(files,target=0.8*20)
filename="dat/cdl_M20.dat"
np.savetxt(filename, np.transpose([rho_m_t, l_t, M_tot_t, M_perp_t, M_para_t, B_tot_t, B_perp_t, B_para_t, va_tot_t, va_perp_t, va_para_t, rho_max_t, M_tot_max_t, B_tot_max_t, va_tot_max_t]))
