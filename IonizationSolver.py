import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class Atom:
    def __init__(self,Z,A,path):
        self.Z=Z
        self.A=A
        self.k = 1.38E-16
        self.m_e = 9.11E-28
        self.c = 3E10
        self.e_e = 4.8E-10
        self.ev_to_erg = 1.6E-12
        self.h = 6.626E-27
        self.m_p = 1.67E-24
        
        self.sigma_0 = 6E-18

        
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
        
    def BB(self,nus=np.logspace(10,20,5000)):
    
        p1 = 2*self.h*nus**3*self.c**-2
        p2 = np.exp(self.h*nus/(self.k*self.temp))-1
        return p1/p2
    
    def photoionCross(self,nu_0,nus=np.logspace(10,20,5000)):
        return np.where(nus>nu_0,self.sigma_0*(nus/nu_0)**-3,0)
        
    def setTemp(self,temp):
        self.temp = temp
        
    def setRho(self,rho):
        self.rho = rho
        
    def setNe(self,ne):
        self.ne = ne
        
    def calcSaha(self):
        E_ion = self.chis[self.level_i] - self.level_E

        r_orb = self.e_e**2/(E_ion*self.ev_to_erg)
        #print(-1*self.n_gas*r_orb**3)
        w = np.exp(-1*self.n_gas*r_orb**3)
        self.lev_n = w*self.level_g*np.exp(-self.level_E*self.ev_to_erg/(self.k*self.temp))

        for i in range(self.n_ions):
            self.ion_part[i] = np.sum(self.lev_n[self.level_i==i])
        lt = self.h**2/(2*np.pi*self.m_e*self.k*self.temp)
        fac = 2/self.ne/lt**1.5
        norm = 1
        for i in np.arange(1,self.n_ions,1):
            currentChi = self.chis[i-1]
            saha = np.exp(-1*currentChi*self.ev_to_erg/(self.k*self.temp))*fac*(self.ion_part[i]/self.ion_part[i-1])
            self.ion_frac[i] = saha*self.ion_frac[i-1]

        norm = np.sum(self.ion_frac)
        self.ion_frac /= norm
        
    def getQNLTERates(self,nus=np.logspace(10,20,5000)):
        self.PI = np.zeros(len(self.level_E))
        BB = self.BB(nus=nus)
        for i in range(len(self.level_E)):
            chi_nu = self.level_E[i]*self.ev_to_erg/self.h
            sigma = self.photoionCross(chi_nu,nus=nus)
            self.PI[i] = 4*np.pi*integrate.trapezoid(BB*sigma/(self.h*nus),nus)
            
    def getQNLTE(self,time,f=1/3):
        self.calcSaha()
        self.getQNLTERates()
        self.omegas = np.zeros(self.n_ions)
        self.LTE_ratios = np.zeros(self.n_ions)
        
        min_rate = 1E-50
        for i in np.arange(1,self.n_ions,1):
            if (self.ion_frac[i] == self.ion_frac[i-1]):
                self.LTE_ratios[i] = 1
            else:
                self.LTE_ratios[i] = self.ion_frac[i]/self.ion_frac[i-1]
            if self.LTE_ratios[i] < min_rate:
                self.LTE_ratios[i] = min_rate
        tmp_ion_frac = np.zeros(self.n_ions)
        tmp_ion_frac[0] = 1.0
        
        for i in np.arange(1,self.n_ions,1):
            bools = np.where(i-1==self.level_i,True,False)
            
            radio = np.sum(self.lev_n[bools]/(self.chis[self.level_i][bools] - self.level_E[bools]))
            
            photo = np.sum(self.PI[bools]*self.lev_n[bools])
            #print(i,self.PI[bools]*self.lev_n[bools])
            
            radio *= self.getRprocHeating(time)*f*self.mu_I*self.m_p
            if photo < min_rate:
                photo = min_rate
            if radio < min_rate:
                self.omegas[i] = 0
            else:
                self.omegas[i] = radio/photo
            #print(i,radio,photo,self.omegas[i])
            tmp_ion_frac[i] = (1+self.omegas[i])*self.LTE_ratios[i]*tmp_ion_frac[i-1]
            tmp_ion_frac = tmp_ion_frac/np.sum(tmp_ion_frac)
        self.ion_frac = tmp_ion_frac
        
        
    def getRprocHeating(self,time):
        epsilon = 8.4939E09*time**(-1.3642)+8.3425E09*np.exp(-time/3.628)+8.8616E08*np.exp(-time/10.847)
        E_beta = 0.2
        A_beta = 1.3E-11
        eta_beta = 2*A_beta/(self.rho*time*60*60*24)
        f_beta = np.log(1+eta_beta)/eta_beta
        return E_beta*f_beta*epsilon
    
        
        
    
