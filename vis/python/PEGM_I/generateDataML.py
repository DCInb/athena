import MassLoss_modified as ml
import numpy as np
import time
timeBegin=time.time()

r_giant = 7.e12
gamma = 5/3

giant = '../../data/planetary_engulfment_test39/'
plot_num = 1000; plot_interv = 1; rc_start = 1.0; rc_num = 19; rc_final=10.0
ml_time_giant_unbound, ml_m_giant_unbound = ml.mass_loss(plot_num, plot_interv, rc_start, rc_num, rc_final,r_giant,gamma, giant,unbound=True)
ml_time_giant_bound, ml_m_giant_bound = ml.mass_loss(plot_num, plot_interv, rc_start, rc_num, rc_final,r_giant,gamma, giant,unbound=False)
ml_time_giant_bound_net, ml_m_giant_bound_net = ml.mass_loss(plot_num, plot_interv, rc_start, rc_num, rc_final,r_giant,gamma, giant,unbound=False,outflow=False)
ml_time_giant_unbound_net, ml_m_giant_unbound_net = ml.mass_loss(plot_num, plot_interv, rc_start, rc_num, rc_final,r_giant,gamma, giant,unbound=True,outflow=False)

# calculate the cumulated mass loss
time_post_giant = np.array(ml_time_giant_bound)
m_dot_time_post_giant = np.array(ml_m_giant_bound)
time_post_giant_unbound = np.array(ml_time_giant_unbound)
m_dot_time_post_giant_unbound = np.array(ml_m_giant_unbound)
time_post_giant_bound_net = np.array(ml_time_giant_bound_net)
m_dot_time_post_giant_bound_net = np.array(ml_m_giant_bound_net)
time_post_giant_unbound_net = np.array(ml_time_giant_unbound_net)
m_dot_time_post_giant_unbound_net = np.array(ml_m_giant_unbound_net)

n_time_giant = np.shape(m_dot_time_post_giant)[0]
n_line_giant = np.shape(m_dot_time_post_giant)[1]
delta_M_giant = np.zeros((n_time_giant,n_line_giant))
delta_M_giant_unbound = np.zeros((n_time_giant,n_line_giant))
delta_M_giant_bound_net = np.zeros((n_time_giant,n_line_giant))
delta_M_giant_unbound_net = np.zeros((n_time_giant,n_line_giant))

for i in range(n_time_giant-1):
    delta_M_giant[i+1,:] = delta_M_giant[i,:]+(m_dot_time_post_giant[i,:]+m_dot_time_post_giant[i+1,:])/2*(time_post_giant[i+1]-time_post_giant[i])
    delta_M_giant_unbound[i+1,:] = delta_M_giant_unbound[i,:]+(m_dot_time_post_giant_unbound[i,:]+m_dot_time_post_giant_unbound[i+1,:])/2*(time_post_giant_unbound[i+1]-time_post_giant_unbound[i])
    delta_M_giant_bound_net[i+1,:] = delta_M_giant_bound_net[i,:]+(m_dot_time_post_giant_bound_net[i,:]+m_dot_time_post_giant_bound_net[i+1,:])/2*(time_post_giant_bound_net[i+1]-time_post_giant_bound_net[i])
    delta_M_giant_unbound_net[i+1,:] = delta_M_giant_unbound_net[i,:]+(m_dot_time_post_giant_unbound_net[i,:]+m_dot_time_post_giant_unbound_net[i+1,:])/2*(time_post_giant_unbound_net[i+1]-time_post_giant_unbound_net[i])
    

filename="PEGMfig/test39_ml.dat"
# Expand time_post_giant to (5, 1) to match others
time_post_giant_expanded = time_post_giant[:, np.newaxis]  # shape (5, 1)

# Concatenate along axis=1
result = np.concatenate(
    [time_post_giant_expanded, delta_M_giant, delta_M_giant_unbound, delta_M_giant_bound_net, delta_M_giant_unbound_net],
    axis=1
)
np.savetxt(filename, result)
print("Total running time is ="+ str(time.time()-timeBegin))