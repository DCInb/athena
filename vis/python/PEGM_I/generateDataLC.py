import light_curve_modified as lc
import numpy as np
import time
timeBegin=time.time()
base_dir = '../../data/planetary_engulfment_test42/'

time_lc, int_lum, radii_aver,temp_aver,rho_aver=lc.light_curve_radial_parallel_plot(1000,1,800,base_dir)

filename="PEGMfig/test42_lc_ana.dat"
np.savetxt(filename, np.transpose([time_lc, int_lum, radii_aver,temp_aver,rho_aver]))
print("Total running time is ="+ str(time.time()-timeBegin))
timeBegin=time.time()
time_lc, int_lum, radii_aver,temp_aver,rho_aver=lc.light_curve_radial_parallel_plot(1000,1,800,base_dir,opac_model='mesa_exe')

filename="PEGMfig/test42_lc_mesa.dat"
np.savetxt(filename, np.transpose([time_lc, int_lum, radii_aver,temp_aver,rho_aver]))
print("Total running time is ="+ str(time.time()-timeBegin))