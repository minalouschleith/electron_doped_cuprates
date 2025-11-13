import numpy as np
from cuprates import *


pars = {} 
pars['a'] = 1.0
pars['Nx'] = 10
pars['Ny'] = 10 
pars['tp'] = -0.35      #next-nearest neighbor hopping 
pars['m'] = 1.0         #to do
pars['c'] = c           #Goldstone velocity from HF 
pars['omega_s'] = Ws    #rescaling factor from HF 
pars['Delta'] = 0.5     # study regime \Delta < 1 
pars['mu'] = 0.0        #chemical potential
pars['g_spin'] = g_spin  

k = 10 
px=0
py=0
model = fermion_spinon_model(pars) 
model.do_ED(np.array([px,py]),k)

pathname = '/Users/mschleith/PhD/Cuprates/results' 
np.save(pathname + '/Nx_4_Ny_4_px_0_py_ED_energies.npy',model.vals)
np.save(pathname + '/Nx_4_Ny_4_px_0_py_ED_states.npy',model.vecs) 
