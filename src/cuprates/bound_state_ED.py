import numpy as np
import scipy as sp
import time 

class fermion_spinon_model(object):

    def __init__(self,pars):
        self.a = pars['a']              #lattice constant
        self.Nx = pars['Nx']            #sites
        self.Ny = pars['Ny']
        self.N = pars['Nx']*pars['Ny']
        self.tp = pars['tp']            #ratio t_2/t_1 with t_1 NN hopping, t_2 NNN hopping
        self.m = pars['m']              #Goldstone mass
        self.c = pars['c']              #Goldstone velocity
        self.chi = pars['chi']          #Goldstone stiffness (?)
        self.Delta = pars['Delta']      #fermion gap
        chi = pars['chi']
        Delta = pars['Delta']
        c = pars['c']
        self.chi_tilde = 2*self.chi*self.Delta**2*self.c**2  #renormalized gap
        self.mu = pars['mu']            #chemical potential
        Qx = int(np.floor(self.Nx/2))
        Qy = int(np.floor(self.Ny/2))
        self.Q = np.array([Qx,Qy])      #AFM ordering vector
        self.mp =  pars['Nx']*pars['Ny']       #regularization of IR divergence 

    def dispersion(self,p):
        return 2*(np.cos(p[0]*self.a)+np.cos(p[1]*self.a)) + 4*self.tp*np.cos(p[0]*self.a)*np.cos(p[1]*self.a)

    def omega(self,p):
        return np.sqrt(p[0]**2+p[1]**2+self.m**2)
    
    def V(self,p): 
        return (24*np.pi*self.m)/(p[0]**2+p[1]**2+self.mp**2)
    
    def omega_denominator(self,p1,p2): 
        return 1/(2*self.chi_tilde*np.sqrt(self.omega(p1)*self.omega(p2)))
    
    #to do 
    def g_vertex(self,p): 
        return 1

    class hopping:
        def __init__(self,ids,ele,flavors):
            #([id1,id2],ele,[flavor(id1),flavor(id2)])
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
        return np.array([i,j],int) 
    
    def linear_id_map(self,p):
        return p[0]*self.Nx+p[1]*self.Ny 
    
    def total_momentum(self,ps):
        p_out = np.zeros(2,int)
        for p in ps: 
            p_out += p
        p_out[0] = np.mod(p_out[0],self.Nx)
        p_out[1] = np.mod(p_out[1],self.Ny)  
        return p_out 

    def make_Kets(self,q):
        Kets = []
        print('building fermion-spinon bound states...')
        start = time.time()
        for id1 in range(self.N): #fermion
            p1 = self.momentum_map(id1)
            # flavors: [f(+), f(-), b(+), b(-), a(+), a(-)] 
            # '-1' = unoccupied; id1, id2 label the single particle momenta of the bound state particles. 
            for id2 in range(self.N): #spinon/anti-spinon
                p2 = self.momentum_map(id2) 
                p_tot = self.total_momentum([p1,p2])

                if np.array_equal(p_tot,q): 
                    Kets.append(np.array([id1, -1, id2, -1, -1, -1])) #f(+)b(+)
                    Kets.append(np.array([id1, -1, -1, id2, -1, -1])) #f(+)b(-)
                    Kets.append(np.array([-1, id1, -1, -1, id2, -1])) #f(-)a(+)
                    Kets.append(np.array([-1, id1, -1, -1, -1, id2])) #f(-)a(-)
        self.Kets = Kets 
        self.Hildim = len(self.Kets) #Hilbert space dimension
        end=time.time()
        print(f'Execution time: {end - start:.4f} seconds')

    def make_hoplist(self):
        '''
        A^dagger(p1) B(p2) 
        '''
        print('making list of 2-particle terms')
        start=time.time()
        hoplist = []
        for id1 in range(self.N):
            p1 = self.momentum_map(id1) 
            for id2 in range(self.N):
                p2 = self.momentum_map(id2) 
                if id1 == id2: 
                    ele = self.dispersion(p1) 
                    hoplist.append(self.hopping([id1,id2],ele,[0,0])) #+f(+)f(+)
                    hoplist.append(self.hopping([id1,id2],ele,[1,1])) #-f(-)f(-)
                    ele = self.omega(p1) 
                    hoplist.append(self.hopping([id1,id2],ele,[2,2])) #+b(+)b(+)
                    hoplist.append(self.hopping([id1,id2],ele,[3,3])) #+b(-)b(-)
                    hoplist.append(self.hopping([id1,id2],ele,[4,4])) #+a(+)a(+)
                    hoplist.append(self.hopping([id1,id2],ele,[5,5])) #+a(-)a(-)
                if np.array_equal(self.total_momentum([p1,p2]),self.Q):
                    ele = -self.Delta 
                    hoplist.append(self.hopping([id1,id2],ele,[0,0])) #+f(+)f(+)
                    hoplist.append(self.hopping([id1,id2],-ele,[1,1])) #-f(-)f(-) 
        self.hoplist = hoplist 
        end=time.time()
        print(f'Execution time: {end - start:.4f} seconds')


    def make_scatter_list(self): 
        '''
        A^dagger(p1) B^dagger(p2) C(p3) D(p4)
        '''
        print('making list of 4-particle terms')
        start=time.time()
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
                        if np.array_equal(self.total_momentum([p1,p2]),self.total_momentum([p3,p4])):
                            #[id3,id4], [id1,id2], ele, [flavor(p3),flavor(p4)], [flavor(p1),flavor(p2)] 
                            #flavors: [f(+), f(-), b(+), b(-), a(+), a(-)] 
                            p_rel_1 = self.total_momentum([p1,p2])
                            p_rel_2 = self.total_momentum([p1,-p4]) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(p_rel_1)*omega_den,[1,5],[0,2]))        #f(+)^dagger b(+)^dagger f(-) a(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(p_rel_1)*omega_den,[1,4],[0,3]))       #f(+)^dagger b(-)^dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(p_rel_2)*omega_den,[1,4],[0,3]))      #f(+)^dagger b(-)^dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum([p1,-p4]))*omega_den,[1,5],[0,2]))       #f(+)^dagger b(+)^dagger f(-) a(-)

                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum([p1,-p4]))*omega_den,[0,2],[1,5]))      #f(-)^dagger a(-)^dagger f(+) b(+) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum([p1,-p4]))*omega_den,[0,3],[1,4]))       #f(-)^dagger a(+)^dagger f(+) b(-) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.dispersion(self.total_momentum([p1,p2]))*omega_den,[0,3],[1,4]))       #f(-)^dagger a(+)^dagger f(+) b(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(self.total_momentum([p1,p2]))*omega_den,[0,2],[1,5]))        #f(-)^dagger a(-)^dagger f(+) b(+)

                            #electromagnetic interaction 
                            #fermions spinons 
                            p = self.total_momentum([p1,-p3])
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [0,4], [0,4])) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], -self.V(p), [1,4], [1,4]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p),[0,5], [0,5])) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], -self.V(p), [1,5], [1,5]))

                            # fermions anti-spinons 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [0,2], [0,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], -self.V(p), [1,2], [1,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [0,3], [0,3]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], -self.V(p), [1,3], [1,3]))

                            # fermions fermions 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [0,0], [0,0]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [0,0], [1,1]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [1,1], [0,0]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [1,1], [1,1])) 

                            # spinons spinons
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [4,4], [4,4]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p), [4,4], [5,5]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p),[5,5],[4,4]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2], self.V(p),[5,5],[5,5]))

                            # anti spinons anti spinons
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.V(p),[2,2],[2,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.V(p),[2,2],[3,3]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.V(p),[3,3],[2,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.V(p),[3,3],[3,3]))

                            # spinons anti spinons 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[4,4],[2,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[2,2],[5,5]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[5,5],[2,2]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[3,3],[4,4]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[4,4],[3,3]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[3,3],[5,5]))
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[5,5],[3,3]))                 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.V(p),[2,2],[4,4]))

                        if np.array_equal(self.total_momentum([p1,p2]),self.total_momentum([p3,p4,self.Q])):
                            
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum([p1,p2]))*omega_den,[1,5],[0,2]))        #f(+)^dagger b(+)^dagger f(-) a(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum([p1,p2]))*omega_den,[1,4],[0,3]))       #f(+)^dagger b(-)^dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum([p1,-p4]))*omega_den,[1,4],[0,3]))      #f(+)^dagger b(-)^dagger f(-) a(+)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.dispersion(-self.total_momentum([p1,-p4]))*omega_den,[1,5],[0,2]))       #f(+)^dagger b(+)^dagger f(-) a(-)

                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum([p1,-p4]))*omega_den,[0,2],[1,5]))      #f(-)^dagger a(-)^dagger f(+) b(+) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum([p1,-p4]))*omega_den,[0,3],[1,4]))       #f(-)^dagger a(+)^dagger f(+) b(-) 
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],-self.g_vertex(-self.total_momentum([p1,p2]))*omega_den,[0,3],[1,4]))       #f(-)^dagger a(+)^dagger f(+) b(-)
                            scatter_list.append(self.scattering([id3,id4],[id1,id2],self.g_vertex(-self.total_momentum([p1,p2]))*omega_den,[0,2],[1,5]))        #f(-)^dagger a(-)^dagger f(+) b(+)

        self.scatter_list = scatter_list 
        end=time.time()
        print(f'Execution time: {end - start:.4f} seconds')


    def do_single_hop(self,Ket,hop):
        Bra = Ket 
        #hop.ids, hop.ele, hop.flavors 
        #Ket: [f(+), f(-), b(+), b(-), a(+), a(-)]
        #hop: ([id1(final),id2(initial)],ele,[flavor(id1),flavor(id2)])
        
        if Ket[hop.flavors[1]] == hop.ids[1] and Ket[hop.flavors[0]] == -1:  #hopping possible?
            Bra[hop.flavors[0]] = hop.ids[0]
            Bra[hop.flavors[1]] = -1
            return Bra
        else:
            return None

    def do_single_scatter(self,Ket,scatter):
        Bra = Ket 
        # scatter.in_ids, scatter.out_ids, scatter.ele, scatter.in_flavors, scatter.out_flavors
        if Ket[scatter.in_flavors[0]] == scatter.in_ids[0] and Ket[scatter.in_flavors[1]] == scatter.in_ids[1] and Ket[scatter.out_flavors[0]] == -1 and Ket[scatter.out_flavors[1]] == -1: #scattering possible?
            Bra[scatter.in_flavors[0]] = -1
            Bra[scatter.in_flavors[1]] = -1
            Bra[scatter.out_flavors[0]] = scatter.out_ids[0]
            Bra[scatter.out_flavors[1]] = scatter.out_ids[1] 
            return Bra
        else:
            return None 
    
    def find_state(self,state_out):
        for (index,Ket) in enumerate(self.Kets):
            if np.array_equal(Ket,state_out):
                return index
        

    def build_Fock(self): 
        out_ids = [] 
        in_ids = []
        Eles = [] 
        for (index_in,state_in) in enumerate(self.Kets): 
            for hop in self.hoplist: 
                state_out = self.do_single_hop(state_in,hop)
                if state_out is None:
                    continue
                else:
                    index_out = self.find_state(state_out)
                    print(index_out)

                    in_ids.append(index_in)
                    out_ids.append(index_out)
                    Eles.append(hop.ele)
            for scatter in self.scatter_list:
                state_out = self.do_single_scatter(state_in,scatter) 
                if state_out is None:
                    continue
                else: 
                    index_out = self.find_state(state_out)
                    in_ids.append(index_in)
                    out_ids.append(index_out)
                    Eles.append(scatter.ele)
        self.in_ids = in_ids
        self.out_ids = in_ids
        self.Eles = Eles 


    def do_ED(self,q,k):
        start_tot=time.time()
        self.make_Kets(q)
        self.make_hoplist()
        self.make_scatter_list()
        print('building sparse matrix')
        start=time.time()
        self.build_Fock()
        H = sp.sparse.coo_matrix((self.Eles, (self.out_ids,self.in_ids)), shape=(self.Hildim,self.Hildim))
        end=time.time()
        print(f'Execution time: {end - start:.4f} seconds')    
        print('diagonalizing Hamiltonian')
        start=time.time()
        vals, vecs = sp.sparse.linalg.eigsh(H, k=k, which='SM')
        self.vals=vals
        self.vecs=vecs
        end=time.time()
        end_tot=time.time()
        print(f'Execution time: {end - start:.4f} seconds')  
        print(f'total runtime {end_tot - start_tot:.4f} seconds')  
        
    