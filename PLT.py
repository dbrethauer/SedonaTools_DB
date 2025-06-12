import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import math

class PLT:
    def __init__(self,path,levels=False):
        self.path = path
        self.files = os.listdir(path)
        self.files = [file for file in self.files if 'plt' == file[:3] and '.h5' == file[-3:]]
        self.files.sort()
        self.files = self.files[1:] #removes plt_00000.h5 since it doesn't contain info
        self.atoms = {}
        self.times = np.array([])
        self.rho = np.array([])
        self.T_gas = np.array([])
        self.T_rad = np.array([])
        self.opacity = np.array([])
        self.Jnu = np.array([])
        self.emissivity = np.array([])
        self.nu = np.array([])
        self.ne = np.array([])
        self.dnu = np.array([])
        self.r = np.array([])
        self.r_inner = np.array([])
        self.lengths = np.array([])
        self.levels = {}
        self.levels_bool = levels
        if len(self.files) > 0 and 'plt_00001.h5' in self.files:
            with h5py.File(path+'/plt_00001.h5','r') as f:
                self.n_zones = len(np.array(f['T_gas']))
                self.nu = np.array(f['nu'])
                self.dnu = np.array([self.nu[i]-self.nu[i-1] if i>0 else self.nu[0] for i in range(len(self.nu))])
                for ky in list(f['zonedata/0'].keys()):
                    if ky[0] == 'Z':
                        self.atoms[ky] = np.array([])
                        if levels:
                            self.levels[ky] = np.array([])

        else:
            print('Warning: No plt files detected!')
    def bb(self,T):
        c=3E10
        h=6.626E-27
        k = 1.38E-16
        p1 = 3*np.log10(2*h*self.nu)-2*np.log10(c)
        p2 = np.log10(np.exp(h*self.nu/(k*T))-1)
        return 10**(p1-p2)
    def extract_times(self):

        self.T_gas = np.empty((self.n_zones,len(self.files)))
        self.T_rad = np.empty((self.n_zones,len(self.files)))
        self.rho = np.empty((self.n_zones,len(self.files)))
        self.r = np.empty((self.n_zones,len(self.files)))
        self.r_inner = np.empty(len(self.files))
        self.lengths = np.empty((self.n_zones,len(self.files)))
        self.ne = np.empty((self.n_zones,len(self.files)))
        for q in range(len(self.files)):
            with h5py.File(self.path+self.files[q],'r') as f:
                self.times = np.append(self.times,np.array(f['time'])/(3600*24),axis=0)
                self.T_gas[:,q] = np.array(f['T_gas'])
                self.T_rad[:,q] = np.array(f['T_rad'])
                self.ne[:,q] = np.array(f['n_elec'])
                self.rho[:,q] = np.array(f['rho'])
                self.r[:,q] = np.array(f['r'])
                self.r_inner[q] = np.array(f['r_inner'])
                temp_r = self.r[:,q]
                temp_r = np.insert(temp_r,0,self.r_inner[q])
                self.lengths[:,q] = self.r[:,q]-temp_r[:-1]
                    
    def extract_ions(self):
        if len(self.times) == 0:
            self.extract_times()
        for ky in self.atoms:
            self.atoms[ky] = np.empty((self.n_zones,len(self.times),int(ky[2:])+1))
            
        self.opacity = np.empty((self.n_zones,len(self.times),len(self.nu)))
        self.Jnu = np.empty((self.n_zones,len(self.times),len(self.nu)))
        self.emissivity = np.empty((self.n_zones,len(self.times),len(self.nu)))
        for q in range(len(self.files)):
            #print(self.files[q])
            with h5py.File(self.path+'/'+self.files[q],'r') as f:
                for z in range(self.n_zones):
                    
                    self.opacity[z,q,:] = np.array(f['zonedata/'+str(z)+'/opacity'])
                    self.Jnu[z,q,:] = np.array(f['zonedata/'+str(z)+'/Jnu'])
                    self.emissivity[z,q,:] = np.array(f['zonedata/'+str(z)+'/emissivity'])
                    for ky in self.atoms:
                        #print(ky,self.times[q],z)
                        if q == 0 and z == 0 and self.levels_bool:
                            n_levels = len(np.array(f['zonedata/'+str(z)+'/'+ky+'/level_fraction']))

                            self.levels[ky] = np.empty((self.n_zones,len(self.times),n_levels))
                        
                            
                        #indices are by zone, by time, then by ion
                        self.atoms[ky][z,q,:] = np.array(f['zonedata/'+str(z)+'/'+ky+'/ion_fraction'])
                        if self.levels_bool:
                            self.levels[ky][z,q,:] = np.array(f['zonedata/'+str(z)+'/'+ky+'/level_fraction'])
                        
    def calcOpticalDepth(self,lamnu,unit):
        if self.opacity.size == 0:
            self.extract_ions()
        if unit == 'AA':
            currentNu = 3E18/lamnu
        elif unit == 'nm':
            currentNu = 3E17/lamnu
        elif unit == 'micron':
            currentNu = 3E14/lamnu
        elif unit == 'nu':
            currentNu = lamnu
        else:
            raise Exception('Unit not currently supported, exiting.')
        nuInd = np.searchsorted(self.nu,currentNu)
        taus = np.flip(self.rho*self.opacity[:,:,nuInd]*self.lengths,axis=0)
        opticalDepths = np.cumsum(taus,axis=0)
        indices = self.n_zones-1-np.argmax(opticalDepths>=1,axis=0)
        fixed_indices = np.where(indices==(self.n_zones-1),0,indices)
        return fixed_indices
        
    def plotIonStageAtom(self,Z=60,ion=0,log=True,s=65,figsize=(16,12)):

        plt.figure(figsize=figsize)
        if log:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,norm=plt.Normalize(vmin=-5,
                                                                vmax=0))
        else:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,norm=plt.Normalize(vmin=0,
                                                                vmax=1))
        ky = 'Z_'+str(Z)
        if log:
            currentData = np.log10(self.atoms[ky][:,:,ion])
        else:
            currentData = self.atoms[ky][:,:,ion]
        currentData = np.where(currentData==np.nan,-5,currentData)
        for q in range(len(self.times)):
            plt.scatter(self.times[q]*np.ones(self.n_zones),np.linspace(0,self.n_zones-1,self.n_zones),color=sm.to_rgba(currentData[:,q]),s=s,marker='s')

        plt.ylim(-0.1,self.n_zones+0.1)

        plt.ylabel('Zone Number',fontsize=14)
        plt.title('Ionization Stage '+ str(ion) + ' for Z = '+ky[-2:],fontsize=18)
        plt.colorbar(sm,location='right',label=r'log$_{10}$(Ion Fraction)')
        
    def plotRho(self,s=65,figsize=(16,12)):

        plt.figure(figsize=figsize)
        currentData = np.log10(self.rho)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,norm=plt.Normalize(vmin=min(np.min(currentData),-20),
                                                                            vmax=max(np.max(currentData),-5)))
        for q in range(len(self.times)):
            plt.scatter(self.times[q]*np.ones(self.n_zones),np.linspace(0,self.n_zones-1,self.n_zones),color=sm.to_rgba(currentData[:,q]),s=s,marker='s')

        plt.ylim(-0.1,self.n_zones+0.1)
        plt.ylabel('Zone Number',fontsize=14)
        plt.title('Density',fontsize=18)
        plt.colorbar(sm,location='right',label=r'log$_{10}$(Density $\frac{g}{cm^3}$)')
        
    def plotTgas(self,s=65,figsize=(16,12)):

        plt.figure(figsize=figsize)
        currentData = np.log10(self.T_gas)
    
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,norm=plt.Normalize(vmin=min(np.min(currentData),2),
                                                                   vmax=min(np.max(currentData),5)))
        for q in range(len(self.times)):
            plt.scatter(self.times[q]*np.ones(self.n_zones),np.linspace(0,self.n_zones-1,self.n_zones),color=sm.to_rgba(currentData[:,q]),s=s,marker='s')

        plt.ylim(-0.1,self.n_zones+0.1)
        plt.ylabel('Zone Number',fontsize=14)
        plt.title('Gas Temperature',fontsize=18)
        plt.colorbar(sm,location='right',label=r'log$_{10}$(Temperature/K)')
        
    def plotGeneric(self,Z,log=False,colormin=0,colormax=10,label='Unlabeled',s=65,figsize=(16,12)):
        plt.figure(figsize=figsize)
        currentData = Z
        if log:
            currentData = np.log10(currentData)
    
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,norm=plt.Normalize(vmin=colormin,
                                                                   vmax=colormax))
        for q in range(len(self.times)):
            plt.scatter(self.times[q]*np.ones(self.n_zones),np.linspace(0,self.n_zones-1,self.n_zones),color=sm.to_rgba(currentData[:,q]),s=s,marker='s')

        plt.ylim(-0.1,self.n_zones+0.1)
        plt.ylabel('Zone Number',fontsize=14)
        #plt.title('Gas Temperature',fontsize=18)
        plt.colorbar(sm,location='right',label=label)
        
    def plotPhotosphere(self,lamnu,unit):

        currentData = self.calcOpticalDepth(lamnu=lamnu,unit=unit)
    
        plt.scatter(self.times,currentData,color='black',s=50)

    #def calcPlanckOpacity(self):
    def plotAllIons(self,Z=60,max_ion=3,s=65,figsize=(12,12)):
        ky = 'Z_'+str(Z)
        allTimes = np.ones((self.n_zones,len(self.times)))
        allZones = np.ones((currentKN.n_zones,len(currentKN.times)))
        zones = np.linspace(0,currentKN.n_zones-1,currentKN.n_zones)
        for q in range(len(allTimes)):
            allTimes[q] = self.times
        for q in range(np.shape(allZones)[1]):
            allZones[:,q] = zones
        colorsSchemes = [plt.cm.Greys,plt.cm.Blues,plt.cm.Oranges,plt.cm.Greys]
        for i in range(max_ion):
            sm = plt.cm.ScalarMappable(cmap=colorsSchemes[i],norm=plt.Normalize(vmin=0,
                                                                vmax=1))
            currentBools = np.where(self.atoms[ky][:,:,i]==np.max(self.atoms[ky],axis=2),True,False)
    #print(np.shape(currentBools))
            currentData = self.atoms[ky][:,:,i][currentBools].flatten()
    #print(np.shape(currentData))
            plt.scatter(allTimes[currentBools],allZones[currentBools].flatten(),color=sm.to_rgba(currentData),s=s,marker='s')
    
    
        
