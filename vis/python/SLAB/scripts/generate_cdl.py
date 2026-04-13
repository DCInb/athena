import sys
sys.path.append("../../")
import cdl
import numpy as np

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="Directory that contains files to plot", required=True)
args = parser.parse_args()

base_dir =  args.dir
name = base_dir.strip('/').split('/')[-1]
match = re.search(r"M([0-9.]+)", base_dir)
if match:
    mach_number = float(match.group(1))
files = [f"{base_dir}/COLL.out1.{str(i).zfill(5)}.athdf" for i in range(1, 51)]

rho_m_t, l_t, M_tot_t, M_perp_t, M_para_t, B_tot_t, B_perp_t, B_para_t, va_tot_t, va_perp_t, va_para_t, rho_max_t, M_tot_max_t, B_tot_max_t, va_tot_max_t, cov_t, M_wei_t_t = cdl.get_cdl_t(files,target=0.8*mach_number)
filename = "../dat/dat_weighted/"+name+".dat"
np.savetxt(filename, np.transpose([rho_m_t, l_t, M_tot_t, M_perp_t, M_para_t, B_tot_t, B_perp_t, B_para_t, va_tot_t, va_perp_t, va_para_t, rho_max_t, M_tot_max_t, B_tot_max_t, va_tot_max_t, cov_t, M_wei_t_t]))
