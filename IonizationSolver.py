import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp

class Atom:
    def __init__(self,Z,A,path):
        self.Z=Z
        self.A=A
        self.k = 1.380658e-16
        self.m_e = 9.10938188e-28
        self.c = 2.99792458e10
        self.e_e = 4.8032068e-10
        self.ev_to_erg = 1.60217646e-12
        self.h = 6.6260755e-27
        self.m_p = 1.67262158e-24
        
        self.sigma_0 = 6.3E-18

        
        self.temp = 1E3
        self.rho = 1E-15
        
        self.ne = 1e5
        self.mu_I = A
        self.n_gas = self.rho/self.mu_I
        
        with h5py.File(path,'r') as f:
            self.chis = np.array(f[str(Z)+'/ion_chi'])
            self.level_E = np.array(f[str(Z)+'/level_E'])
            self.level_g = np.array(f[str(Z)+'/level_g'])
            self.level_i = np.array(f[str(Z)+'/level_i'])
        self.n_ions = len(self.chis)
        self.ion_frac = np.zeros(self.n_ions)
        self.ion_frac[0] = 1.0
        self.ion_part = np.zeros(self.n_ions)
        self.lev_n = np.array(len(self.level_E))
        self.PI = np.zeros(len(self.level_E))
        
    def BB(self,nus=np.logspace(12,17,2000)):
    
        p1 = 2*self.h*nus**3*self.c**-2
        p2 = np.exp(self.h*nus/(self.k*self.temp))-1
        return p1/p2
    
    def photoionCross(self,nu_0,nus=np.logspace(12,17,2000)):
        return np.where(nus>=nu_0,self.sigma_0*(nus/nu_0)**-3,0)
        
    def setTemp(self,temp):
        self.temp = temp
        
    def setRho(self,rho):
        self.rho = rho
        self.n_gas = self.rho/(self.mu_I*self.m_p)
        
    def setNe(self,ne):
        self.ne = ne
        
    def calcSaha(self):
        self.ion_part = np.zeros(self.n_ions)
        self.ion_frac = np.zeros(self.n_ions)
        self.ion_frac[0] = 1.0
        E_ion = self.chis[self.level_i] - self.level_E

        r_orb = self.e_e**2/(E_ion*self.ev_to_erg)
        #print(-1*self.n_gas*r_orb**3)
        w = np.where(E_ion>0,np.exp(-1*self.n_gas*r_orb**3),1)
        #print(w)
        
        self.lev_n = w*self.level_g*np.exp(-self.level_E*self.ev_to_erg/(self.k*self.temp))
        #for i in np.arange(0,self.n_ions,1):
        #    bools = np.where(self.level_i==i,True,False)
        #    norm = np.sum(self.lev_n[bools])
        #    for q in range(len(self.level_E)):
        #        if self.level_i[q] == i:
        #            self.lev_n /= norm
            

        for i in range(self.n_ions):
            self.ion_part[i] = np.sum(self.lev_n[self.level_i==i])
        lt = self.h**2/(2*np.pi*self.m_e*self.k*self.temp)
        fac = 2/self.ne/lt**1.5
        for i in np.arange(1,self.n_ions,1):
            currentChi = self.chis[i-1]
            saha = np.exp(-currentChi*self.ev_to_erg/(self.k*self.temp))
            saha *= fac*(self.ion_part[i]/self.ion_part[i-1])
            self.ion_frac[i] = saha*self.ion_frac[i-1]

        norm = np.sum(self.ion_frac)
        self.ion_frac /= norm
        self.lev_n = self.lev_n/self.ion_part[self.level_i]
        #self.lev_n = np.where(self.lev_n<1E-30,1E-30,self.lev_n)
        
        #self.ion_frac = np.where(self.ion_frac < 1E-30,1E-30,self.ion_frac)
        
    def directRates(self,time,f=1/3):
        self.ion_part = np.zeros(self.n_ions)
        self.ion_frac = np.zeros(self.n_ions)
        self.ion_frac[0] = 1.0
        E_ion = self.chis[self.level_i] - self.level_E

        r_orb = self.e_e**2/(E_ion*self.ev_to_erg)
        #print(-1*self.n_gas*r_orb**3)
        w = np.where(E_ion>0,np.exp(-1*self.n_gas*r_orb**3),1)
        #print(w)
        
        self.lev_n = w*self.level_g*np.exp(-self.level_E*self.ev_to_erg/(self.k*self.temp))

            

        for i in range(self.n_ions):
            self.ion_part[i] = np.sum(self.lev_n[self.level_i==i])
            
        self.radio = np.zeros(self.n_ions)
        self.photo = np.zeros(self.n_ions)
        
        lt = self.h**2/(2*np.pi*self.m_e*self.k*self.temp)
        fac = 2/self.ne/lt**1.5
        
        for i in np.arange(1,self.n_ions,1):
            bools = np.where(i-1==self.level_i,True,False)
            zeta = self.chis[i-1]*self.ev_to_erg/self.k/self.temp
            radio = np.sum(self.lev_n[bools]/(self.chis[self.level_i][bools] - self.level_E[bools]))
            
            photo = np.sum(self.PI[bools]*self.lev_n[bools])# + coll_ion
            #print(i,self.PI[bools]*self.lev_n[bools])
            
            radio *= self.getRprocHeating(time)*f*self.mu_I*self.m_p/self.ev_to_erg
            
            self.radio[i] = radio
            self.photo[i] = photo
            
            currentChi = self.chis[i-1]
            
            saha = np.exp(-currentChi*self.ev_to_erg/(self.k*self.temp))
            saha *= fac*(self.ion_part[i]/self.ion_part[i-1])
            self.ion_frac[i] = saha*self.ion_frac[i-1]
        
        norm = np.sum(self.ion_frac)
        self.ion_frac /= norm
        self.lev_n = self.lev_n/self.ion_part[self.level_i]
        
    def getQNLTERates(self,nus=np.logspace(12,17,2000)):
        self.PI = np.zeros(len(self.level_E))
        BB = self.BB(nus=nus)
        allSigmas = np.empty((len(self.level_E),len(nus)))
        n_eff = np.sqrt(self.chis[self.level_i]/(self.chis[self.level_i]-self.level_E)) #Effecive excitation quantum number
        cs_factor = (self.level_i.astype(float)+1)**-2
        for i in range(len(self.level_E)):
            chi_nu = self.level_E[i]*self.ev_to_erg/self.h
            allSigmas[i,:] = n_eff[i]*cs_factor[i]*self.photoionCross(chi_nu,nus=nus)

        self.PI = 4*np.pi*integrate.trapezoid(BB*allSigmas/(self.h*nus),nus)
        
            
    def getQNLTE(self,time,f=1/3):
        self.calcSaha()
        self.getQNLTERates()
        self.omegas = np.zeros(self.n_ions)
        self.LTE_ratios = np.zeros(self.n_ions)
        self.radio = np.zeros(self.n_ions)
        self.photo = np.zeros(self.n_ions)
        self.recomb = np.zeros(self.n_ions)
        recomb_constant = 3E-11*((self.temp/1E4)**-0.75)*self.ne#-(1/3)*(self.temp/1E4)**-0.5)*self.ne
        
        min_rate = 1E-50
        for i in np.arange(1,self.n_ions,1):

            rad_recom = i**2*recomb_constant
            self.recomb[i] = rad_recom
            if (self.ion_frac[i] == self.ion_frac[i-1]):
                self.LTE_ratios[i] = min_rate #should only happen when they are both the zero
            else:
                self.LTE_ratios[i] = self.ion_frac[i]/self.ion_frac[i-1]
            if self.LTE_ratios[i] < min_rate:
                self.LTE_ratios[i] = min_rate
        tmp_ion_frac = np.zeros(self.n_ions)
        tmp_ion_frac[0] = 1.0
        
        for i in np.arange(1,self.n_ions,1):
            bools = np.where(i-1==self.level_i,True,False)
            zeta = self.chis[i-1]*self.ev_to_erg/self.k/self.temp
            #coll_ion = 2.7*(zeta)**-2*((self.temp)**-1.5)*np.exp(-zeta)*self.ne
            #print(zeta**-2,self.temp**-1.5,np.exp(-zeta),zeta)
            #print((self.chis[self.level_i][bools] - self.level_E[bools]))
            radio = np.sum(self.lev_n[bools]/(self.chis[self.level_i][bools] - self.level_E[bools]))
            
            
            photo = np.sum(self.PI[bools]*self.lev_n[bools])# + coll_ion
            #print(i,self.PI[bools]*self.lev_n[bools])
            
            radio *= self.getRprocHeating(time)*f*self.mu_I*self.m_p/self.ev_to_erg
            
            self.radio[i] = radio
            self.photo[i] = photo
            
            #lt = self.h**2/(2*np.pi*self.m_e*self.k*self.temp)
            #fac = 2/lt**1.5
            #currentChi = self.chis[i-1]
            #saha = np.exp(-currentChi*self.ev_to_erg/(self.k*self.temp))
            #saha *= fac*(self.ion_part[i]/self.ion_part[i-1])
            
            
            
            min_ion_rate = 1E-50#self.recomb[i]*saha
            #print(min_ion_rate,self.photo[i],self.temp)
            if photo < min_ion_rate:
                photo = min_ion_rate
                #self.omegas[i] = (self.radio[i])/self.recomb[i]
                #self.LTE_ratios[i] = -1.0
            if radio < min_rate:
                self.omegas[i] = 0
            else:
                self.omegas[i] = radio/photo
            #print(i,radio,photo,self.omegas[i])
            if self.LTE_ratios[i] != -1.0:
                tmp_ion_frac[i] = (1+self.omegas[i])*self.LTE_ratios[i]*tmp_ion_frac[i-1]
            else:
                tmp_ion_frac[i] = self.omegas[i]*tmp_ion_frac[i-1]
            
            tmp_ion_frac = tmp_ion_frac/np.sum(tmp_ion_frac)
            
        self.ion_frac = tmp_ion_frac
        self.lev_n = np.where(self.lev_n<1E-30,1E-30,self.lev_n)
        #print(self.omegas)
        
        
    def getRprocHeating(self,time):
        epsilon = 8.4939E09*time**(-1.3642)+8.3425E09*np.exp(-time/3.628)+8.8616E08*np.exp(-time/10.847)
        E_beta = 0.2
        A_beta = 1.3E-11
        eta_beta = 2*A_beta/(self.rho*time*60*60*24)
        f_beta = np.log(1+eta_beta)/eta_beta
        return E_beta*f_beta*epsilon
        
    def calcNe(self,max_ion=1):
        total_ne = 0
        for i in range(len(max_ions)):
            total_ne += i*self.ion_frac[i]*self.n_gas
        return total_ne
        
    def solveSingleIon(self,time,f=1/3):
        self.getQNLTERates()
        self.calcSaha()
        
        self.radio = np.zeros(self.n_ions)
        self.photo = np.zeros(self.n_ions)
        self.recomb = np.zeros(self.n_ions)
        alpha = 3E-11*((self.temp/1E4)**-0.75)#-(1/3)*(self.temp/1E4)**-0.5)*self.ne
        
        
        bools = np.where(0==self.level_i,True,False)
        #zeta = self.chis[0]*self.ev_to_erg/self.k/self.temp
        #print(np.shape(bools),np.shape(self.lev_n),np.shape(self.chis[self.level_i]))

        radio = np.sum(self.lev_n[bools]/(self.chis[self.level_i][bools] - self.level_E[bools]))
            
            
        photo = np.sum(self.PI[bools]*self.lev_n[bools])# + coll_ion
            
            
        radio *= self.getRprocHeating(time)*f*self.mu_I*self.m_p/self.ev_to_erg
            
        self.radio[1] = radio
        self.photo[1] = photo
        
        self.ion_frac = np.zeros(self.n_ions)
        self.ion_frac[0] = 1
        
        eta = (radio+photo)/alpha
        sol = eta/2+np.sqrt(eta*(eta-4))/2#,eta/2-np.sqrt(eta*(eta-4))/2)
        
        self.ion_frac[1] = sol
        self.ion_frac /= np.sum(self.ion_frac)
        self.recomb[1] = alpha*self.ion_frac[1]*self.n_gas
        #self.ion_frac[1] =
        
    def solveDoubleIon(self,time,f=1/3,eta0_val=None,eta1_val=None):
        
        self.calcSaha()
        
        self.ion_frac = np.zeros(self.n_ions)
        
        self.radio = np.zeros(self.n_ions)
        self.photo = np.zeros(self.n_ions)
        self.recomb = np.zeros(self.n_ions)
        alpha = 3E-11*((self.temp/1E4)**-0.75)
        
        radio_const = self.getRprocHeating(time)*f*self.mu_I*self.m_p/self.ev_to_erg
        
        if eta0_val == None and eta1_val == None:
            self.getQNLTERates()
        
            for i in [1,2]:
            #print(i)
                bools = np.where(i-1==self.level_i,True,False)
            #zeta = self.chis[0]*self.ev_to_erg/self.k/self.temp
        #print(np.shape(bools),np.shape(self.lev_n),np.shape(self.chis[self.level_i]))

                radio = np.sum(self.lev_n[bools]/(self.chis[self.level_i][bools] - self.level_E[bools]))
            
            
                photo = np.sum(self.PI[bools]*self.lev_n[bools])# + coll_ion
            
            
                radio *= radio_const
                self.radio[i] = radio
                self.photo[i] = photo
        
            eta0_val = (self.photo[1]+self.radio[1])/(alpha)
            eta1_val = (self.photo[2]+self.radio[2])/(alpha*2**2)
        
        func = sp.Symbol('f', real=True)
     # Define polynomial coefficients
        coeffs = [1,eta0_val,eta0_val * (eta1_val - 1),-2 * eta0_val * eta1_val]
    # Create polynomial and get numeric roots
        poly = sp.Poly.from_list(coeffs, gens=func)
        roots = poly.nroots()

    # Filter real positive roots
        real_roots = [r for r in roots if sp.re(r) > 0 and abs(sp.im(r)) < 1e-10]

        if not real_roots:
            print("No positive real root found.")
            return None

        fe = float(real_roots[0]) # take smallest positive root
        denom = 1 + eta0_val/fe + (eta0_val * eta1_val)/fe**2
        f0 = 1 / denom
        f1 = (eta0_val / fe) * f0
        f2 = (eta0_val * eta1_val)/fe**2 * f0
        
        self.ion_frac[:3] = np.array([f0,f1,f2])
        self.setNe(fe*self.n_gas)

        #return {"fe": fe,"f0": f0,"f1": f1,"f2": f2}

        
    def brent(self,func,*args,max_iters=100,aa=1,bb=10,epsilon=0.01):
        n = 0
        a = aa
        b = bb
        fa = func(*args)
        fb = func(*args)
        
        if (fa*fb >= 0):
            return a
        if (np.abs(fa) < np.abs(fb)):
            temp = a
            a = b
            b = temp
            temp = fa
            fa = fb
            fb = temp
        c = a
        fc = fa
        running = 1
        bisect = True
        bdelta = 1E-15*np.min(np.abs(fa),np.abs(fb))
        s = 0
        d = 0
        while running:
            if (fa != fc and fb != fc):
                s = a*fb*fc/(fa-fb)/(fa-fc) + b*fc*fa/(fb-fc)/(fb-fa)+c*fa*fb/(fc-fa)/(fc-fb)
            else:
                s = b-fb*(b-a)/(fb-fa)
            
            #cond1 = (s < (3*a+b)/4 and s < b) or (s > (3*a+b)/4 and s> b)
            #cond2 = bisect and np.abs(s-b) >= np.abs(b-c)/2
            #cond3 = !bisect and (np.abs(s-b) >= np.abs(c-d)/2)
            #cond4 = bisect and (np.abs(b-c) < bdelta)
            #cond5 = !bisect and (np.abs(c-d) < bdelta)
            
            #if cond1 or cond2 or don3 or cond4 or cond5:
            #    s = (a+b)/2
            #    bisect = True
            #else
            #    bisect = False
        
    
        
        
    
