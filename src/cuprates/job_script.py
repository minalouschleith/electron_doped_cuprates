import numpy as np
import matplotlib.pyplot as plt
from cuprates import *

pars = {} 
pars['a'] = 1.0
pars['Nx'] = 3
pars['Ny'] = 3
pars['tp'] = -0.35      #next-nearest neighbor hopping 
pars['m'] = 1.0         #to do
pars['c'] = 0.988       #Goldstone velocity
pars['chi'] = 1.0       #to do
pars['Delta'] = 1.0     #to do
pars['mu'] = 0.0        #chemical potential

k = 10 
px=0
py=0
model = fermion_spinon_model(pars) 
model.do_ED(np.array([px,py]),k)

pathname = '/Users/mschleith/PhD/Cuprates/results' 
np.save(pathname + '/Nx_4_Ny_4_px_0_py_ED_energies.npy',model.vals)
np.save(pathname + '/Nx_4_Ny_4_px_0_py_ED_states.npy',model.vecs) 
