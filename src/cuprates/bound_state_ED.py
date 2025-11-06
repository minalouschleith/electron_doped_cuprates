import numpy as np 

class fermion_spinon_model(object):

    def __init__(self,pars):
        self.a = pars['a']              #lattice constant
        self.Nx = pars['Nx']            #sites
        self.Ny = pars['Ny']
        self.N = self.Nx*self.Ny 
        self.tp = pars['tp']            #ratio t_2/t_1 with t_1 NN hopping, t_2 NNN hopping
        self.m = pars['m']              #Goldstone mass
        self.c = pars['c']              #Goldstone velocity
        self.chi = pars['chi']          #Goldstone stiffness (?)
        self.Delta = pars['Delta']      #fermion gap
        self.chi_tilde = 2*self.chi*self.Delta**2*self.c**2  #renormalized gap
        self.mu = pars['mu']            #chemical potential
        self.Q = [np.pi, np.pi]/self.a  #AFM ordering vector

    def dispersion(self,p):
        return 2*(np.cos(p[0]*self.a)+np.cos(p[1]*self.a)) + 4*self.tp*np.cos(p[0]*self.a)*np.cos(p[1]*self.a)

    def omega(self,p):
        return np.sqrt(p[0]**2+p[1]**2+self.m**2)
    
    def V(self,p): 
        return 1/(p[0]**2+p[1]**2)
    
    def omega_denominator(self,p1,p2): 
        return 1/(2*self.chi_tilde*np.sqrt(self.omega(p1[0],p1[1])*self.omega(p2[0],p2[1])))
    
    #to do 
    def g_vertex(self,p): 
        return 1

    class hopping:
        def __init__(self,ids,ele,flavors):
            self.ids = ids
            self.ele = ele 
            self.flavors = flavors
    
    class scattering:
        def __init__(self,in_ids,out_ids,ele,in_flavors,out_flavors):
            self.in_ids = in_ids
            self.out_ids = out_ids
            self.ele = ele
            self.in_flavors = in_flavors 
            self.out_flavors = out_flavors

    class Fock:
        def __init__(self,Kets,Bras,Hops):
            self.Kets = Kets
            self.Bras = Bras
            self.Hops = Hops

    def momentum_map(self,id):

        '''

        [0,0] [0,1] ... [0,Nx] 
        [1,0] ...       [1,Nx]
        .       .            .
        .        [i,j]     .
        .         .          .
        [Ny,0]...       [Nx,Ny]

        -> [px,py]
        '''

        i = id%self.Nx       #quotient
        j = id//self.Nx      #remainder
        return np.array([i,j]) 
    
    def linear_id_map(self,p):
        return p[0]*self.Nx+p[1]*self.Ny 
    
    def total_momentum(self,p1,p2):
        px_tot = (p1[0]+p2[0]) % self.Nx
        py_tot = (p1[1]+p2[1]) % self.Ny
        return np.array([px_tot,py_tot])

    def make_Kets(self,q):
        self.q = q                      #bound state momentum
        N = self.N                      #Nx*Ny
        Nx = self.Nx
        Ny = self.Ny 
        Psis = []

        for id1 in range(N): #fermion
            p1 = self.momentum_map(id1)
            # flavors: [f(+), f(-), b(+), b(-), a(+), a(-)] 
            for id2 in range(N): #spinon/anti-spinon
                p2 = self.momentum_map(id2) 
                p_tot = self.total_momentum(p1,p2)
                if p_tot == q: 
                    Psis.append(np.array([id1, -1, id2, -1, -1, -1])) #f(+)b(+)
                    Psis.append(np.array([id1, -1, -1, id2, -1, -1])) #f(+)b(-)
                    Psis.append(np.array([-1, id1, -1, -1, id2, -1])) #f(-)a(+)
                    Psis.append(np.array([-1, id1, -1, -1, -1, id2])) #f(-)a(-)

        self.Kets = Psis 
        self.Hildim = len(self.Kets) #Hilbert space dimension

    def make_hoplist(self):
        '''
        A^\dagger(p1) B(p2) 
        '''
        hoplist = []
        
        for id1 in range(self.N):
            p1 = self.momentum_map(id1) 
            for id2 in range(self.N):
                p2 = self.momentum_map(id2) 
                ele = -1j*self.V(self.total_momentum(p1,p2)) 
                hoplist.append(self.hopping([id1,id2],ele,[0,0])) #+f(+)f(+)
                hoplist.append(self.hopping([id1,id2],-ele,[2,2])) #-f(-)f(-)
                hoplist.append(self.hopping([id1,id2],ele,[2,2])) #+b(+)b(+)
                hoplist.append(self.hopping([id1,id2],ele,[3,3])) #+b(-)b(-)
                hoplist.append(self.hopping([id1,id2],-ele,[4,4])) #-a(+)a(+)
                hoplist.append(self.hopping([id1,id2],-ele,[5,5])) #-a(-)a(-)
                if id1 == id2: 
                    ele = self.dispersion(p1[0],p1[1]) 
                    hoplist.append(self.hopping([id1,id2],ele,[0,0])) #+f(+)f(+)
                    hoplist.append(self.hopping([id1,id2],ele,[1,1])) #-f(-)f(-)
                    ele = self.omega(p1[0],p2[0]) 
                    hoplist.append(self.hopping([id1,id2],ele,[2,2])) #+b(+)b(+)
                    hoplist.append(self.hopping([id1,id2],ele,[3,3])) #+b(-)b(-)
                    hoplist.append(self.hopping([id1,id2],ele,[4,4])) #+a(+)a(+)
                    hoplist.append(self.hopping([id1,id2],ele,[5,5])) #+a(-)a(-)
                if  P_tot == self.Q:
                    ele = -self.Delta 
                    hoplist.append(self.hopping([id1,id2],ele,[0,0])) #+f(+)f(+)
                    hoplist.append(self.hopping([id1,id2],-ele,[1,1])) #-f(-)f(-) 

    return hoplist 
                
    def make_scatter_list(self):

        '''
        A^\dagger(p1) B^\dagger(p2) C(p3) D(p4)
        '''

        scatter_list = []

        for id1 in range(self.N):
            p1 = self.momentum_map(id1) #outgoing 
            for id2 in range(self.N):
                p2 = self.momentum_map(id2) #outgoing
                for id3 in range(self.N):
                    p3 = self.momentum_map(id3) #incoming
                    for id4 in range(self.N):
                        p4 = self.momentum_map(id4) #incoming 
                        omega_den = self.omega_denominator(p2,p4)
                        if self.total_momentum(p1,p2) == self.total_momentum(p3,p4): #momentum conservation 
                            #[id3,id4], [id1,id2], ele, [flavor(p3),flavor(p4)], [flavor(p1),flavor(p2)] 
                            #flavors: [f(+), f(-), b(+), b(-), a(+), a(-)] 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum(p1,p2))*omega_den,[1,5],[0,2]))        #f(+)^\dagger b(+)^\dagger f(-) a(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum(p1,p2))*omega_den,[1,4],[0,3]))       #f(+)^\dagger b(-)^\dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum(p1,-p4))*omega_den,[1,4],[0,3]))      #f(+)^\dagger b(-)^\dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum(p1,-p4))*omega_den,[1,5],[0,2]))       #f(+)^\dagger b(+)^\dagger f(-) a(-)

                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum(p1,-p4))*omega_den,[0,2],[1,5]))      #f(-)^\dagger a(-)^\dagger f(+) b(+) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum(p1,-p4))*omega_den,[0,3],[1,4]))       #f(-)^\dagger a(+)^\dagger f(+) b(-) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum(p1,p2))*omega_den,[0,3],[1,4]))       #f(-)^\dagger a(+)^\dagger f(+) b(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum(p1,p2))*omega_den,[0,2],[1,5]))        #f(-)^\dagger a(-)^\dagger f(+) b(+)
                        
                        if self.total_momentum(p1,p2) == self.total_momentum(self.total_momentum(p3,p4),self.Q):
                            
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum(p1,p2))*omega_den,[1,5],[0,2]))        #f(+)^\dagger b(+)^\dagger f(-) a(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum(p1,p2))*omega_den,[1,4],[0,3]))       #f(+)^\dagger b(-)^\dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum(p1,-p4))*omega_den,[1,4],[0,3]))      #f(+)^\dagger b(-)^\dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(-self.total_momentum(p1,-p4))*omega_den,[1,5],[0,2]))       #f(+)^\dagger b(+)^\dagger f(-) a(-)

                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum(p1,-p4))*omega_den,[0,2],[1,5]))      #f(-)^\dagger a(-)^\dagger f(+) b(+) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum(p1,-p4))*omega_den,[0,3],[1,4]))       #f(-)^\dagger a(+)^\dagger f(+) b(-) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum(p1,p2))*omega_den,[0,3],[1,4]))       #f(-)^\dagger a(+)^\dagger f(+) b(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum(p1,p2))*omega_den,[0,2],[1,5]))        #f(-)^\dagger a(-)^\dagger f(+) b(+)

        return scatter_list

    def do_hop(self,Ket):
        bra = []
        return bra 


