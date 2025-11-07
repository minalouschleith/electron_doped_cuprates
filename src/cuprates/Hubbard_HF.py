import numpy as np
import scipy as sp

class Hubbard_hartree_fock(object):
    """

    """

    def __init__(self, pars):

        self.tp = pars['tp']
        self.D = pars['D']
        self.U = pars['U']
        self.Nx = pars['Nx']
        assert self.Nx%2 == 0
        self.Ny = pars['Ny']
        assert self.Ny%2 == 0
        self.kx = -np.pi + np.arange(self.Nx)/self.Nx*2.*np.pi
        self.ky = -np.pi + np.arange(self.Ny)/self.Ny*2.*np.pi
        self.kxs,self.kys = np.meshgrid(self.kx,self.ky,indexing='ij')
        self.int_type = pars['int_type']
        self.Ne = pars['Ne']
        assert self.Ne > 0
        assert self.Ne < (2*self.Nx*self.Ny)

        self.HF_basis = False

    def roll_q(self,a,qx,qy):
        a = np.roll(a,qx,axis=0)
        #shift tuple 'a' with qx along kx-axis
        a = np.roll(a,qy,axis=1)
        #shift tuple 'a' with qy along ky-axis
        return a

    def compute_Vq(self): #Compute Fourier transform of interaction

        V = np.zeros((self.Nx,self.Ny),dtype=complex)

        if self.int_type == "Hubbard": #local Hubbard interaction
            return V+self.U

        if self.int_type == "screened_Coulomb": #screened Coulomb V(r) = U e^(-r/D)/(r+1)

            if self.Nx == self.Ny:
                R = int(self.Nx/2) + 2
            if self.Nx > self.Ny:
                R = int(self.Ny/2) + 2
            if self.Ny > self.Nx:
                R = int(self.Nx/2) + 2

            for i in range(2*R):
                for j in range(2*R):
                    x = -R + i
                    y = -R + j
                    r = np.sqrt(x**2+y**2)
                    V += self.U*np.exp(-r/self.D)/(r+1)*np.exp(1j*x*self.kxs)*np.exp(1j*y*self.kys)
            return V.real

    def HF_init(self, rho, q = np.array([0.,0.]), translation = True, measure_O = False):

        self.q = q
        self.translation = translation
        self.measure_O = measure_O

        dispersion_up = -2*(np.cos(self.kxs-self.q[0]/2)+np.cos(self.kys-self.q[1]/2)) - self.tp*2*(np.cos(self.kxs+self.kys-self.q[0]/2-self.q[1]/2)+np.cos(self.kxs-self.kys-self.q[0]/2+self.q[1]/2))
        dispersion_down = -2*(np.cos(self.kxs+self.q[0]/2)+np.cos(self.kys+self.q[1]/2)) - self.tp*2*(np.cos(self.kxs+self.kys+self.q[0]/2+self.q[1]/2)+np.cos(self.kxs-self.kys+self.q[0]/2-self.q[1]/2))
        self.h0 = np.zeros((self.Nx,self.Ny,2,2),dtype=complex)
        self.h0[:,:,0,0] = dispersion_up
        self.h0[:,:,1,1] = dispersion_down

        self.V = self.compute_Vq()

        if self.translation == False:
            assert self.q[0] == 0. and self.q[1] == 0.
            h0 = np.zeros((self.Nx,self.Ny,2,self.Nx,self.Ny,2),dtype=complex)
            for i in range(self.Nx):
                for j in range(self.Ny):
                    h0[i,j,:,i,j,:] = self.h0[i,j,:,:]
            self.h0 = h0


        self.rho = rho
        G = self.compute_G(rho)
        self.HF = self.h0 + G

        self.stats = {'E': []}
        self.SB = {'translation': []}



    def compute_G(self,rho):

        HH = self.compute_HH(rho)
        HF = self.compute_HF(rho)

        return HH + HF


    def compute_HH(self,rho): #Compute Hartree Hamiltonian from density matrix rho

        if self.translation == True:

            HH = np.zeros((self.Nx,self.Ny,2,2),dtype=complex)
            HH[:,:,0,0] = self.Ne/(self.Nx*self.Ny)*self.V[int(self.Nx/2),int(self.Ny/2)]
            HH[:,:,1,1] = self.Ne/(self.Nx*self.Ny)*self.V[int(self.Nx/2),int(self.Ny/2)]

            return HH

        if self.translation == False:

            Id = np.eye(self.Nx*self.Ny*2,dtype=complex)
            Id = np.reshape(Id,(self.Nx,self.Ny,2,self.Nx,self.Ny,2))
            #generate identity matrix with indices (ks,k's') -> if k=k' and s=s' element = 1

            HH = np.zeros((self.Nx,self.Ny,2,self.Nx,self.Ny,2),dtype=complex)

            for i in range(self.Nx):
                for j in range(self.Ny):

                    qx = -int(self.Nx/2) + i
                    qy = -int(self.Ny/2) + j

                    rho_q = self.roll_q(rho,qx,qy)
                    #shift rho_(ks;ks') -> rho_(k-q,s;k,s')
                    rho_q = np.reshape(rho_q,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
                    rho_q = np.diag(rho_q)
                    #take the diagonal elements (spin up, spin up) or (spin down, spin down)


                    HH += self.V[i,j]*np.sum(rho_q)*self.roll_q(Id,-qx,-qy)/(self.Nx*self.Ny)
                    #the shifted Identity matrix represents I_(k+q,s;k,s')

            return HH

    def compute_HF(self,rho): #Compute Fock Hamiltonian from density matrix rho

        if self.translation == True:

            HF = np.zeros((self.Nx,self.Ny,2,2),dtype=complex)

            for i in range(self.Nx):
                for j in range(self.Ny):
                    qx = -int(self.Nx/2) + i
                    qy = -int(self.Ny/2) + j
                    HF += -self.V[i,j]*self.roll_q(rho,qx,qy)/(self.Nx*self.Ny)
                #VRAAG: som van mogelijkse energie bijdragen door (-2,-2), (-2,-1), (-2,0), (-2,1), (-1,-2), ... hoppings 
            return HF


        if self.translation == False:

            HF = np.zeros((self.Nx,self.Ny,2,self.Nx,self.Ny,2),dtype=complex)

            for i in range(self.Nx):
                for j in range(self.Ny):
                    qx = -int(self.Nx/2) + i
                    qy = -int(self.Ny/2) + j

                    rho_q = self.roll_q(rho,qx,qy)
                    rho_q = np.transpose(rho_q,(3,4,5,0,1,2))
                    rho_q = self.roll_q(rho_q,qx,qy)
                    rho_q = np.transpose(rho_q,(3,4,5,0,1,2))

                    HF += -self.V[i,j]*rho_q/(self.Nx*self.Ny)

            return HF


    def compute_E(self,rho): #Compute energy from density matrix rho

        G = self.compute_G(rho)

        if self.translation == True:

            E = np.einsum('ijst,ijts->',rho,self.h0+G/2)
            return E

        if self.translation == False:

            rho = np.reshape(rho,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            h0 = np.reshape(self.h0,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            G = np.reshape(G,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            E = np.trace(np.dot(rho,h0+G/2))
            return E


    def compute_rho(self,HF): #Compute density matrix rho from mean-field Hamiltonian HF

        if self.translation == True:

            E,U = np.linalg.eigh(HF)
            args = np.argsort(E, axis=None)[:self.Ne]
            P = np.zeros(E.shape,dtype=complex)
            P = np.reshape(P,(self.Nx*self.Ny*2,))
            P[args] = 1.
            P = np.reshape(P,(self.Nx,self.Ny,2))

            rho = np.einsum('ijst, ijt->ijst', U,  P)
            rho = np.einsum('ijst, ijrt->ijsr', rho, U.conj())

            return rho

        if self.translation == False:

            HF = np.reshape(HF,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            E,U = np.linalg.eigh(HF)
            args = np.argsort(E)[:self.Ne]
            P = np.zeros((self.Nx*self.Ny*2,),dtype=complex)
            P[args] = 1.
            P = np.diag(P)

            rho = np.dot(U,P)
            rho = np.dot(rho,np.conj(U.T))
            rho = np.reshape(rho,(self.Nx,self.Ny,2,self.Nx,self.Ny,2))

            return rho


    def ODA_step(self):
        """
            Updates energy and runs one step of ODA. Returns dE
        """
        h0 = self.h0
        HF = self.HF
        rho = self.rho

        E = self.compute_E(rho)
        self.stats['E'].append(E/(self.Nx*self.Ny))


        if self.measure_O:
            self.record_order_params(rho)

        rhon = self.compute_rho(HF)
        Gn = self.compute_G(rhon)
        HFn = h0 + Gn

        drho = rhon - rho

        #E(rho + l drho) = E + s l + c l^2 / 2
        if self.translation == True:
            s =  np.einsum('ijst, ijts', drho, HF).real
            c =  np.einsum('ijst, ijts', drho, HFn).real - s
        if self.translation == False:
            drho_r = np.reshape(drho,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            HF_r = np.reshape(HF,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))
            HFn_r = np.reshape(HFn,(self.Nx*self.Ny*2,self.Nx*self.Ny*2))

            s = np.trace(np.dot(drho_r,HF_r)).real
            c = np.trace(np.dot(drho_r,HFn_r)).real - s

        if s > 0:
            print("s, c:", s, c)
            #raise RuntimeWarning
            return 0.

        if c <= 0:
            l = 1.
        else:
            l = - s/c
        if l > 1:
            l = 1.


        self.rho = rho + l*drho
        self.HF = (1 - l)*HF + l*HFn

        return -l*s - l*l*c/2.


    def record_order_params(self,rho):
        O = self.compute_Ordparams(rho)
        self.SB['translation'].append(O)


    def compute_Ordparams(self,rho): #Compute translation symmetry breaking order parameter

        if self.translation == False:
            rho = np.einsum('ijsmnt,ijsmnt->ijmn',rho,np.conj(rho))
            rho = np.sqrt(np.abs(rho))
            rho = np.reshape(rho,(self.Nx*self.Ny,self.Nx*self.Ny))
            rho = rho - np.diag(np.diag(rho))

            return np.linalg.norm(rho)

        if self.translation == True:
            return None

    def compute_charge_density(self): #Compute real-space charge density of rho

        assert self.translation == False

        rho = self.rho[:,:,0,:,:,0] + self.rho[:,:,1,:,:,1]
        rho = np.reshape(rho,(self.Nx*self.Ny,self.Nx*self.Ny))

        rx = np.arange(self.Nx)
        ry = np.arange(self.Ny)
        rxs,rys = np.meshgrid(rx,ry,indexing='ij')
        rxs = np.reshape(rxs,(self.Nx*self.Ny,))
        rys = np.reshape(rys,(self.Nx*self.Ny,))
        kxs = np.reshape(self.kxs,(self.Nx*self.Ny,))
        kys = np.reshape(self.kys,(self.Nx*self.Ny,))

        U = np.exp(1j*(np.outer(rxs,kxs) + np.outer(rys,kys)))/np.sqrt(self.Nx*self.Ny)

        rho = np.dot(np.dot(U,rho),np.conj(U.T))

        return np.reshape(np.diag(rho),(self.Nx,self.Ny))

    def compute_spin_density(self): #Compute real-space spin density of rho

        if self.translation == True:

            rho = self.rho
            rho = np.reshape(rho,(self.Nx*self.Ny,2,2))
            rho = np.sum(rho,axis=0)/(self.Nx*self.Ny)

            rx = np.arange(self.Nx)
            ry = np.arange(self.Ny)
            rxs,rys = np.meshgrid(rx,ry,indexing='ij')

            sx = rho[0,1]*np.exp(1j*(self.q[0]*rxs+self.q[1]*rys)) + rho[1,0]*np.exp(-1j*(self.q[0]*rxs+self.q[1]*rys))
            sy = 1j*(rho[0,1]*np.exp(1j*(self.q[0]*rxs+self.q[1]*rys)) - rho[1,0]*np.exp(-1j*(self.q[0]*rxs+self.q[1]*rys)))
            sz = np.zeros((self.Nx,self.Ny),dtype=complex) + (rho[0,0] - rho[1,1])

            return sx, sy, sz

        if self.translation == False:

            sx = self.rho[:,:,0,:,:,1] + self.rho[:,:,1,:,:,0]
            sy = 1j*(self.rho[:,:,0,:,:,1] - self.rho[:,:,1,:,:,0])
            sz = self.rho[:,:,0,:,:,0] - self.rho[:,:,1,:,:,1]
            sx = np.reshape(sx,(self.Nx*self.Ny,self.Nx*self.Ny))
            sy = np.reshape(sy,(self.Nx*self.Ny,self.Nx*self.Ny))
            sz = np.reshape(sz,(self.Nx*self.Ny,self.Nx*self.Ny))

            rx = np.arange(self.Nx)
            ry = np.arange(self.Ny)
            rxs,rys = np.meshgrid(rx,ry,indexing='ij')
            rxs = np.reshape(rxs,(self.Nx*self.Ny,))
            rys = np.reshape(rys,(self.Nx*self.Ny,))
            kxs = np.reshape(self.kxs,(self.Nx*self.Ny,))
            kys = np.reshape(self.kys,(self.Nx*self.Ny,))

            U = np.exp(1j*(np.outer(rxs,kxs)+np.outer(rys,kys)))/np.sqrt(self.Nx*self.Ny)

            sx = np.dot(np.dot(U,sx),np.conj(U.T))
            sy = np.dot(np.dot(U,sy),np.conj(U.T))
            sz = np.dot(np.dot(U,sz),np.conj(U.T))

            return np.reshape(np.diag(sx),(self.Nx,self.Ny)), np.reshape(np.diag(sy),(self.Nx,self.Ny)), np.reshape(np.diag(sz),(self.Nx,self.Ny))




    def goto_HF_basis(self): #Go to HF basis to do the collective mode calculation

        assert self.translation == True

        self.E, U = np.linalg.eigh(self.HF)
        self.U = U

        args = np.argsort(self.E, axis=None)[:self.Ne]
        P = np.zeros(self.E.shape)
        P = np.reshape(P,(self.Nx*self.Ny*2,))
        P[args] = 1.
        P = np.reshape(P,(self.Nx,self.Ny,2))
        self.rho_diag = P

        self.F = np.zeros((self.Nx,self.Ny,self.Nx,self.Ny,2,2),dtype=complex) #Form factors

        for i in range(self.Nx):
            for j in range(self.Ny):

                qx = -int(self.Nx/2) + i
                qy = -int(self.Ny/2) + j

                Uq = self.roll_q(U,qx,qy)

                self.F[i,j,:,:,:,:] = np.einsum('ijsb,ijsa->ijba',np.conj(Uq),U)

        self.HF_basis = True


    def constr_hash_table(self,rho_diag,qx,qy): #Construct hash table which stores which combinations of \varphi_{\alpha\beta} are nonzero, i.e. for which n_\alpha != n_\beta

        assert self.translation == True

        rho = rho_diag
        rho_q = self.roll_q(rho,qx,qy)

        hash_table = np.zeros((self.Nx,self.Ny,2,2),dtype=int)

        Z = []
        index = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                for s in range(2):
                    for t in range(2):
                        l = int(rho[i,j,s]+1e-8)
                        r = int(rho_q[i,j,t]+1e-8)
                        if l == r:
                            hash_table[i,j,s,t] = -1
                        else:
                            hash_table[i,j,s,t] = index
                            index += 1
                            if r == 1:
                                Z.append(1.)
                            if r == 0:
                                Z.append(-1.)

        Z = np.array(Z)
        Z = np.diag(Z)

        return hash_table, Z

    def compute_TDHFspectrum_exact(self,qx,qy):

        assert self.translation == True

        if not self.HF_basis:
            self.goto_HF_basis()

        rho = self.rho_diag
        rho_q = self.roll_q(rho,qx,qy)
        E = self.E
        E_q = self.roll_q(self.E,qx,qy)
        F = self.F

        hash_table, Z = self.constr_hash_table(rho,qx,qy)
        size = Z.shape[-1]

        #Construct contribution of HF dispersion to collective modes
        MD = np.zeros((size,size),dtype=complex)
        for i in range(self.Nx):
            for j in range(self.Ny):
                for s in range(2):
                    for t in range(2):
                        index = hash_table[i,j,s,t]
                        if index >= 0:
                            MD[index,index] = np.abs(E[i,j,s]-E_q[i,j,t])


        #Hartree contribution
        iq = (qx + int(self.Nx/2))%self.Nx
        jq = (qy + int(self.Ny/2))%self.Ny
        f = F[iq,jq,:,:,:,:]
        vec = np.zeros((size,),dtype=complex)
        for i in range(self.Nx):
            for j in range(self.Ny):
                for s in range(2):
                    for t in range(2):
                        index = hash_table[i,j,s,t]
                        if index >= 0:
                            vec[index] = f[i,j,t,s]
        MH = np.outer(np.conj(vec),vec)*self.V[iq,jq]/(self.Nx*self.Ny)


        #Fock contribution
        MF = np.zeros((size,size),dtype=complex)
        for i in range(self.Nx):
            for j in range(self.Ny):
                fl = F[i,j,:,:,:,:]
                fr = self.roll_q(fl,qx,qy)
                px = -int(self.Nx/2) + i
                py = -int(self.Ny/2) + j
                for a in range(self.Nx):
                    for b in range(self.Ny):
                        for st in range(4):
                            for ru in range(4):
                                s = st%2
                                t = int( ((st-s)/2)%2 )
                                r = ru%2
                                u = int( ((ru-r)/2)%2 )
                                ap = (a-px)%self.Nx
                                bp = (b-py)%self.Ny
                                index1 = hash_table[ap,bp,s,t]
                                index2 = hash_table[a,b,r,u]
                                if index1 >=0 and index2 >= 0:
                                    MF[index2,index1] += -np.conj(fl[a,b,s,r])*fr[a,b,t,u]*self.V[i,j]/(self.Nx*self.Ny)

        M = np.dot(Z,MD + MH + MF)
        Ecm,Ucm = np.linalg.eig(M)

        return Ecm,Ucm, hash_table, Z
