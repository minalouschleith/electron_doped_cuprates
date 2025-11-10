import numpy as np
import scipy as sp
import time as time 
from .Hubbard_HF import *

def get_vertices(pars): 
    print('getting vertices')
    HFock = Hubbard_hartree_fock(pars) 
    #Initial seed for HF
    rho0 = np.zeros((HFock.Nx,HFock.Ny,2,2),dtype=complex)
    rho0[:,:,0,0] = 1.
    RM = np.random.random(rho0.shape)-0.5 #+ 1j*(np.random.random(rho0.shape)-0.5)
    RM = RM + np.transpose(np.conj(RM),(0,1,3,2))
    Es,V = np.linalg.eigh(RM)
    rho0 = np.einsum('ijmn,ijno->ijmo',rho0,V)
    rho0 = np.einsum('ijmn,ijmo->ijno',np.conj(V),rho0)
    N = np.trace(rho0,axis1=2,axis2=3)
    rho0 = rho0/np.sum(N)*HFock.Ne

    HFock.HF_init(rho0, q = np.array([np.pi*(1-0.),np.pi*(1-0.)]), translation = True, measure_O = True)
    dEs = []    
    E0,U0 = np.linalg.eigh(HFock.h0)
    for j in range(500):
        dE = HFock.ODA_step()
        dEs.append(dE) 
    E,U = np.linalg.eigh(HFock.HF)
    