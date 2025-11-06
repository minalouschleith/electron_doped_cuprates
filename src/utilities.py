import numpy as np
#import Hubbard_HF 

sig_0 = np.array([[1,0], [0,1]])
sig_x = np.array([[0, 1], [1, 0]])
sig_y = np.array([[0, -1j], [1j, 0]])
sig_z = np.array([[1, 0], [0, -1]])

def dispersion(px, py, tp, a):
    return 2*(np.cos(px*a)+np.cos(py*a)) + 4*tp*np.cos(px*a)*np.cos(py*a)

def omega(px, py, m):
    return np.sqrt(px**2+py**2+m**2)
    
def V(p): 
    return 1/(p[0]**2+p[1]**2)

