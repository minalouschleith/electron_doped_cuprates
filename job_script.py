
import os 
import numpy as np 
from cuprates import *


print("Current working directory:", os.getcwd())
Ws = np.load('HF_results_for_cluster/10x10/Nx_10_Ny_10_TDHF_factor_Ws.npy')
g_spin = np.load('HF_results_for_cluster/10x10/Nx_10_Ny_10_g_spin_q.npy')
c = np.load('HF_results_for_cluster/10x10/Nx_10_Ny_10_TDHF_velocity.npy')

pars = {} 
pars['a'] = 1.0 
pars['Nx'] = 10 
pars['Ny'] = 10 
pars['tp'] = -0.35 #next-nearest neighbor hopping 
pars['m'] = 1.0 #to do 
pars['c'] = c #Goldstone velocity from HF 
pars['omega_s'] = Ws #rescaling factor from HF 
pars['Delta'] = 0.5 # study regime \Delta < 1 
pars['mu'] = 0.0 #chemical potential 
pars['g_spin'] = g_spin
k = 10 
px=0 
py=0

model = fermion_spinon_model(pars) 
model.do_ED(np.array([px,py]),k)
np.save('/user/gent/505/vsc50528/electron_doped_cuprates/ED_results/14-11-2025_first_runs/Nx_10_Ny_10_energies.npy',model.vals)
np.save('/user/gent/505/vsc50528/electron_doped_cuprates/ED_results/14-11-2025_first_runs/Nx_10_Ny_10_eigenstates.npy',model.vecs)
