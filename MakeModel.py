import numpy as np
import h5py
import matplotlib.pyplot as plt



days = 60*60*24        # days conversion to seconds
fd     = 1 #fraction of
lf = 0.01 #lanthanide fraction

class Model:
    def __init__(self,Z_inc=np.array([1]),A_inc=np.array([1]),time=0.25*days):
        self.temp = np.array([])
        self.rho = np.array([])
        self.Z = Z_inc #Zs must be in increasing order!!!!!!
        self.A = A_inc
        self.erad = np.array([])
        self.comp = np.array([])
        self.X_rproc = np.array([])
        self.time = time
        self.X_lan = np.array([])
        self.lan_ind0 = np.searchsorted(self.Z,58)
        self.lan_ind1 = np.searchsorted(self.Z,71)
        self.m_sun = 1.989*10**33
        self.c = 2.998*10**10
        self.days = 24*60*60
        self.arad   = 7.5657e-15
        self.volumes = np.array([])
        self.singleXlan = None
        self.rho_min = 1E-20
        
    def findXlan(self):
        totalLan = np.sum(self.comp[...,self.lan_ind0:self.lan_ind1],axis=-1)
        total = np.sum(self.comp,axis=-1)
        self.X_lan = totalLan/total
        
    def setXlan(self,Xlan):
        self.findXlan()
        if type(Xlan) == float:
            self.singleXlan = Xlan
            fullZs = np.broadcast_to(self.Z,np.shape(self.rho)+(len(self.Z),))
            fullXlans = np.repeat(self.X_lan[...,np.newaxis],len(self.Z),axis=-1)
            coeffs = np.where((fullZs>=58)&(fullZs<=71),Xlan/fullXlans,(1-Xlan)/(1-fullXlans))
            self.comp = coeffs*self.comp
        elif np.shape(Xlan) == np.shape(self.rho):
            fullZs = np.broadcast_to(self.Z,np.shape(self.rho)+(len(self.Z),))
            fullXlans = np.repeat(self.X_lan[...,np.newaxis],len(self.Z),axis=-1)
            fullXlanGoals = np.repeat(Xlan[...,np.newaxis],len(self.Z),axis=-1)
            coeffs = np.where((fullZs>=58)&(fullZs<=71),fullXlanGoals/fullXlans,(1-fullXlanGoals)/(1-fullXlans))
            self.comp = coeffs*self.comp
            
        else:
            raise Exception("Incompatible Xlan input")
            
        self.findXlan()

        
    def setComp(self,NewXs):
        if np.shape(NewXs) == np.shape(self.Z):
            self.comp[...,:] = NewXs
        elif np.shape(NewXs) == np.shape(self.comp):
            self.comp = NewXs
        else:
            raise Exception("Incompatible composition input")
        self.findXlan()

        
    def setTemp(self,use_rproc=True,temps=None):
        if use_rproc:
            A1 = 8.4939E+09
            alpha = 1.3642E+00
            eint0 =  A1/(2.0-alpha)*(self.time/3600.0/24.0)**(1 - alpha)*3600.0*24.0
            self.temp = (eint0*self.rho/self.arad)**0.25
        else:
            if not np.shape(temps) == np.shape(self.temps):
                raise Exception("Temperature arrays do not match! Try again!")
            self.temp = temps
            
    def setXrproc(self,Xrproc):
        if np.shape(Xrproc) == np.shape(self.rho):
            self.X_rproc = Xrproc
        elif type(Xrproc) == float or float(Xrproc) == int(Xrproc):
            self.X_rproc = Xrproc*np.ones(np.shape(self.rho))
        else:
            raise Exception("Incompatible Xrproc input")
            
    def addDensityProfile(self, rhos, Xs):
        comp_by_rho = np.repeat(rhos[...,np.newaxis],len(self.Z),axis=-1)
        current_comp_by_rho = np.repeat(self.rho[...,np.newaxis],len(self.Z),axis=-1)
        total_comp_by_rho = current_comp_by_rho+comp_by_rho
        
        self.comp = (comp_by_rho*Xs+current_comp_by_rho*self.comp)/(total_comp_by_rho)
        self.rho = rhos+self.rho
        
        self.rho = np.where(self.rho<=2*self.rho_min,self.rho_min,self.rho)
        
        bools = np.where(self.rho==self.rho_min,True,False)
        #print(self.comp[bools])
        self.comp[bools] = self.getNewXlan(self.comp.flatten()[:len(self.Z)],Xlan_goal=1E-9)
        
        self.findXlan()
        
        self.setTemp()
        
    def getNewXlan(self,Xs,Xlan_goal):
        currentXlan = np.sum(Xs[(self.Z>=58)&(self.Z<=71)])/np.sum(Xs)
        coeffs = np.where((self.Z>=58)&(self.Z<=71),Xlan_goal/currentXlan,(1-Xlan_goal)/(1-currentXlan))
        return Xs*coeffs
        
        

class Model1D(Model):
    def __init__(self,n_zone=80,Z_inc=np.array([1]),A_inc=np.array([1]),time=0.25*days,rmin=0):
        super().__init__(Z_inc=Z_inc,A_inc=A_inc,time=time)
        self.temp = np.zeros(n_zone)
        self.rho = np.zeros(n_zone)
        self.comp = np.ones((n_zone,len(self.Z)))
        self.erad = np.zeros(n_zone)
        self.X_rproc = np.zeros(n_zone)
        self.n_zone = n_zone
        self.volume = np.zeros(n_zone)
        self.rmin = rmin
        
    def setProperties(self,mass=None,Xlan=None,vk=None,use_rproc=True):
        self.setMass(mass=mass)
        self.setVel(vk=vk)
        if use_rproc:
            self.setXlan(Xlan=Xlan)
    
    def setMass(self,mass):
        self.mass = mass
        
    def setVel(self,vk):
        self.v = vk
        if vk >= 1:
            print("This is in units of c, did you forget?")
        
    def setVol(self):
        for i in range(self.n_zone):
            if i != 0:
                self.volume[i] = 4/3*np.pi*((self.vx[i]*self.time)**3-(self.vx[i-1]*self.time)**3)
            else:
                self.volume[i] = 4/3*np.pi*((self.vx[i]*self.time)**3-self.rmin**3)
        
    def BrokenPowerLaw(self,mass,velocity,n_inner=1,n_outer=10,use_rproc=True,resetGrid=False):
        if n_inner == 3 or n_outer == 3 or n_inner == 5 or n_outer == 5:
            raise Exception("Warning, value of power law index not valid for this formalism!")
        eta_rho = (4*np.pi*((n_inner-n_outer)/((3-n_inner)*(3-n_outer))))**-1
        eta_v = (((5-n_inner)*(5-n_outer))/((3-n_inner)*(3-n_outer)))**0.5
        
        v_t = eta_v*velocity*self.c
        
        if resetGrid:
            self.vmax = 3*v_t
            while self.vmax > self.c:
                self.vmax /= 1.5
        
            self.rmax = self.time*self.vmax
            self.dr   = self.rmax/(1.0*self.n_zone)
            self.dv   = self.vmax/(1.0*self.n_zone)
            self.vx   = np.arange(self.dv,self.vmax+0.1,self.dv)
        
        self.setVol()
        
        currentRho = eta_rho*(mass*self.m_sun)/(v_t*self.time)**3*np.where(self.vx<v_t,((self.vx-self.dv/2)/v_t)**(-1*n_inner),((self.vx-self.dv/2)/v_t)**(-1*n_outer))
        totalMass = np.sum(currentRho*self.volume)
        currentRho *= (mass*self.m_sun)/totalMass
        
        return currentRho
        #self.setTemp(use_rproc=use_rproc)
        
    def ConstantDensity(self,mass,vmax=0.3,v_cut=0.3,use_rproc=True,resetGrid=False):
        
        if resetGrid:
            self.vmax = vmax*self.c
            if self.vmax > self.c:
                raise Exception("Warning: maximum velocity is greater than c!")
            self.rmax = self.time*self.vmax
            self.dr   = self.rmax/(1.0*self.n_zone)
            self.dv   = self.vmax/(1.0*self.n_zone)
            self.vx   = np.arange(self.dv,self.vmax+0.1,self.dv)
        
        self.setVol()
        
        currentRho = np.where(self.vx<v_cut*self.c,mass*self.m_sun/(4/3*np.pi*self.time*v_cut*self.c)**3*np.ones(self.n_zone),self.rho_min)
        totalMass = np.sum(currentRho[self.vx<v_cut*self.c]*self.volume[self.vx<v_cut*self.c])
        currentRho[self.vx<v_cut*self.c] *= (mass*self.m_sun)/totalMass
        
        return currentRho
        
        #self.setTemp(use_rproc=use_rproc)
        
    def writeh5(self,name,use_rproc=True):
        if use_rproc:
            fout = h5py.File(name + "_"+"{:.0E}".format(self.singleXlan)+'X_lan_' + str(self.v)+"v" +"_" + "{:.1E}".format(self.mass) + 'M' + '_1D.h5','w')
            fout.create_dataset('time',data=[self.time],dtype='d')
            fout.create_dataset('Z',data=self.Z,dtype='i')
            fout.create_dataset('A',data=self.A,dtype='i')
            fout.create_dataset('rho',data=self.rho,dtype='d')
            fout.create_dataset('temp',data=self.temp,dtype='d')
            fout.create_dataset('v',data=self.vx,dtype='d')
            fout.create_dataset('Xrproc',data=self.X_rproc,dtype='d')
            fout.create_dataset('comp',data=self.comp,dtype='d')
            fout.create_dataset('r_out',data=self.vx*self.time,dtype='d')
            fout.create_dataset('r_min',data=[self.rmin],dtype='d')
            fout.create_dataset('erad',data=self.erad,dtype='d')
        else:
            fout = h5py.File(name + "_" + str(self.v)+"v" +"_" + "{:.1E}".format(self.mass) + 'M' + '_1D.h5','w')
            fout.create_dataset('time',data=[self.time],dtype='d')
            fout.create_dataset('Z',data=self.Z,dtype='i')
            fout.create_dataset('A',data=self.A,dtype='i')
            fout.create_dataset('rho',data=self.rho,dtype='d')
            fout.create_dataset('temp',data=self.temp,dtype='d')
            fout.create_dataset('v',data=self.vx,dtype='d')
            fout.create_dataset('Xrproc',data=self.X_rproc,dtype='d')
            fout.create_dataset('comp',data=self.comp,dtype='d')
            fout.create_dataset('r_out',data=self.vx*self.time,dtype='d')
            fout.create_dataset('r_min',data=[self.rmin],dtype='d')
            fout.create_dataset('erad',data=self.erad,dtype='d')
        
        
        
                
    
class Model2D(Model):
    def __init__(self,n_zone=80,Z_inc=np.array([1]),A_inc=np.array([1]),time=0.25*days):
        super().__init__(Z_inc=Z_inc,A_inc=A_inc,time=time)
        n_z = 2*n_zone
        self.temp = np.zeros((n_zone,n_z))
        self.rho = np.zeros((n_zone,n_z))
        self.comp = np.zeros((n_zone,len(self.Z)))
        self.erad = np.zeros(n_zone)
        self.X_rproc = np.zeros(n_zone)

name   = "FreeNeutron_test"    # base name of model


#constants, cgs


#Solar Abundances, log_10(X/H)+12 from Asplund 2009. These are NUMBER desnities, must change to mass for Sedona, EDITING Z=61 to be the same at Z=60
Z_sol =   np.array([1,  2,     3,    4,    5,    6,    7,    8,    9,    10,   11,   12,  13,   14,   15,   16,   17,  18,  19,   20,   21,   22,   23,   24,   25,   26,  27,   28,   29,   30,   31,   32,   36,   37,   38,   39,   40,   41,   42,   44,   45,   46,   47,   49,  50,   54,   56,  57,   58,   59,   60,   61,   62,   63,   64,   65,  66,  67,   68,   69,  70,   71,  72,   74,   76,  77,   79,   81,  82,   90])
A_sol =   np.array([1,  4,     7,    9,    11,   12,   14,   16,   19,   20,   23,   24,  27,   28,   31,   32,   35,  36,  39,   40,   45,   48,   51,   52,   55,   56,  59,   58,   63,   64,   69,   74,   84,   85,   88,   89,   90,   93,   98,   102,  103,  106,  107,  115, 120,  129,  138, 139,  140,  141,  142,  145,  152,  153,  158,  159, 164, 165,  166,  169, 174,  175, 180,  184,  192, 193,  197,  205, 208,  232])
dex_sol = np.array([12, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41, 7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5, 4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58, 1.46, 1.88, 1.75, 0.91, 1.57, 0.94, 0.8, 2.04, 2.24, 2.18, 1.1, 1.58, 0.72, 1.42, 1.4,  0.96, 0.52, 1.07, 0.3, 1.1, 0.48, 0.92, 0.1, 0.84, 0.1, 0.85, 0.85, 1.4, 1.38, 0.92, 0.9, 1.75, 0.02])

Z_meteor = np.array([33,  34,  35,  48,  51,  52,  53,  55,  73,   75,  78,  80,  83,  92])
A_meteor = np.array([75,  79,  80,  112, 122, 128, 127, 133, 181,  186, 195, 201, 209, 238])
dex_meteor=np.array([2.30,3.34,2.54,1.71,1.01,2.18,1.55,1.08,-0.12,0.26,1.62,1.17,0.65,-0.54])

includeMeteor = True

if includeMeteor:
    for i in range(len(Z_meteor)):
        index = np.searchsorted(Z_sol,Z_meteor[i])
        Z_sol = np.insert(Z_sol,index,Z_meteor[i])
        A_sol = np.insert(A_sol,index,A_meteor[i])
        dex_sol = np.insert(dex_sol,index,dex_meteor[i])


#Increasing iron by 100x
#dex_sol[list(Z_sol).index(26)] = dex_sol[list(Z_sol).index(26)]+2

X_sol_to_H = A_sol*10**(dex_sol-12)
X_sol = X_sol_to_H/(sum(X_sol_to_H))
#print(X_sol)

#Solar rProcess Residuals from Simmerer 2004, these are the fraction of each solar element that comes from rProcess- ASSUMING Z=61 to be similar to Z=60!!!!
Z_sol_resid =  np.array([1,6,2,31,   32,   33,   34,   35,   36,  37,    38,   39,   40,   41,   42,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   90,   92])
frac_sol_resid=np.array([1,1,1,0.431,0.516,9.785,0.655,0.833,0.437,0.499,0.110,0.281,0.191,0.324,0.323,0.610,0.839,0.555,0.788,0.499,0.678,0.225,0.839,0.803,0.944,0.796,0.850,0.147,0.246,0.186,0.508,0.421,0.421,0.669,0.973,0.819,0.933,0.879,0.936,0.832,0.829,0.682,0.796,0.510,0.588,0.462,0.911,0.916,0.988,0.949,0.944,0.420,0.341,0.214,0.647,1.000,1.000])

if True:
    logs = np.log10(frac_sol_resid)
    A_sol_resid = np.ones(len(Z_sol_resid))
    dex_sol_resid = np.ones(len(Z_sol_resid))
    for i in range(len(Z_sol_resid)):
        if Z_sol_resid[i] in list(Z_sol):
            current_ind = list(Z_sol).index(Z_sol_resid[i])
            A_sol_resid[i] = A_sol[current_ind]
            dex_sol_resid[i] = dex_sol[current_ind]+logs[i]
        else:
            current_ind = list(Z_meteor).index(Z_sol_resid[i])
            A_sol_resid[i] = A_meteor[current_ind]
            dex_sol_resid[i] = dex_meteor[current_ind]+logs[i]


#Based on HD 222925, Roederer et al 2022
Z_star = np.array([30,   31,   32,   33,   34,   37,   38,   39,   40,   41,   42,   44,   45,   46,   47,   48,   49,   50,   51,   52,   56,   57,   58,   59,   60,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   82])
A_star = np.array([66,   70,   73,   75,   79,   85,   88,   89,   91,   93,   96,   101,  103,  106,  108,  112,  115,  119,  122,  128,  137,  139,  140,  141,  144,  150,  152,  157,  159,  163,  165,  167,  169,  173,  175,  178,  181,  184,  186,  190,  192,  195,  197,  207])
dex_star=np.array([3.15, 1.26, 1.46, 1.01, 2.62, 2.10, 1.98, 1.04, 1.74, 0.71, 1.36, 1.32, 0.64, 1.05, 0.44, 0.34, 0.51, 1.39, 0.37, 1.63, 1.26, 0.51, 0.85, 0.22, 0.88, 0.62, 0.38, 0.82, 0.18, 1.01, 0.12, 0.73,-0.09, 0.55,-0.04, 0.32,-0.30, 0.02, 0.16, 1.17, 1.28, 1.45, 0.53, 1.14])

bad_elems = [3,4,5,9,14,25]

fe = 6
Z_even = np.array([30,   31,   32,   33,   34,   37,   38,   39,   40,   41,   42,   44,   45,   46,   47,   48,   49,   50,   51,   52,   56,   57,   58,   59,   60,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   82])
A_even = np.array([66,   70,   73,   75,   79,   85,   88,   89,   91,   93,   96,   101,  103,  106,  108,  112,  115,  119,  122,  128,  137,  139,  140,  141,  144,  150,  152,  157,  159,  163,  165,  167,  169,  173,  175,  178,  181,  184,  186,  190,  192,  195,  197,  207])
dex_even=np.array([fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   0.85, 0.22, 0.88, 0.62, 0.38, 0.82, 0.18, 1.01, 0.12, 0.73,-0.09, 0.55,-0.04, fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe,   fe])

#filter_elems = [2]


filter_elems =[31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,49,50,51,52,56,58,59,60,61,62,63,64,65,66,67,68,69,70] #Used for Brethauer+24


fake_rproc_elems =      {31:13,    #maximum in cmfgen is Z=28, all elemnts must be that or smaller
			32:14,33:15,34:16,35:17,36:18,37:19,38:20,39:21,40:22,41:23,42:24,43:25,44:26,45:27,46:28,
                        49:13,50:14,51:15,52:16,56:20}

    
def changeXlan(Xs,X_lan_goal):
    currentXlan = findXlan(Xs)
    coeff = np.zeros(len(Z))
    for i in range(len(Z)):
        if Z[i] >= 58 and Z[i] <= 71:
            coeff[i] = X_lan_goal/currentXlan
        else:
            coeff[i] = (1-X_lan_goal)/(1-currentXlan)
    return coeff*Xs
    


def getAbundances(whichZ,whichA,dex,whichElems,convertFromNumDens=True):
    elem_bools = np.zeros(len(whichZ),dtype=bool)
    for i in range(len(whichZ)):
        if whichZ[i] in whichElems:
            elem_bools[i] = 1
    Z_short = whichZ[elem_bools]
    A_short = whichA[elem_bools]
    X_short = dex[elem_bools]
    if convertFromNumDens == True:
        X_to_H = A_short*10**(X_short-12)
        X_short = X_to_H/sum(X_to_H)
    return Z_short, A_short, X_short

def enhanceElement(Z,whichZs,Xs,factor):
    coeffs = np.ones(len(whichZs))
    currentVal = Xs[list(whichZs).index(Z)]
    currentXlan = findXlan(Xs)
    newVal = currentVal*factor
    if Z >= 58 and Z <= 71:
        for i in range(len(whichZs)):
            if whichZs[i] == Z:
                coeffs[i] = factor
            elif whichZs[i] >= 58 and whichZs[i] <= 71:
                if currentXlan < newVal:
                     print('Warning: Factor would increase element above lanthanide fraction. Reduce factor and rerun model')
                coeffs[i] = (currentXlan-newVal)/(currentXlan-currentVal)
    else:
        for i in range(len(whichZs)):
            if whichZs[i] == Z:
                coeffs[i] = factor
            elif whichZs[i] < 58 or whichZs[i] > 71:
                coeffs[i] = (1-currentXlan-newVal)/(1-currentXlan-currentVal)
    print("Enhancing element "+str(Z)+ " by a factor of " + str(factor))
    return coeffs*Xs


atomicData = getAbundances(Z_sol_resid,A_sol_resid,dex_sol_resid,filter_elems)
#print(atomicData[2])

#total_MW_Zs, total_MW_As, total_MW_Xs = getAbundances(Z_sol_resid,A_sol_resid,dex_sol_resid,np.ones(len(Z_sol_resid),dtype=bool)) 
#print(Z_sol_resid,A_sol_resid,dex_sol_resid)
#print('Milky Way Mass Fraction ',total_MW_Xs)
#Xs = enhanceElement(48,list(atomicData[0]),list(atomicData[2]),10)

Z, A, Xs = atomicData
#A = list(atomicData[1])
#Xs = list(atomicData[2])
#print(Xs)

use_fake_rproc = False
if use_fake_rproc:
    for i in range(len(Z)):
        if Z[i] in fake_rproc_elems.keys():
            Z[i] = fake_rproc_elems[Z[i]]



setEven = False

if setEven:
    currentLan = findXlan(Xs)
    total_not = 0
    for i in range(len(Z)):
        if (Z[i] < 58) or (Z[i] > 71):
            total_not += 1
    for i in range(len(Z)):
        if Z[i] < 58 or Z[i] > 71:
            Xs[i] = (1-currentLan)/total_not 

#################################################################################
#This will overide the above composition code

#Z = [14,15,16,21,22,23,24,25,26, 27, 28, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
#A = [80,80,80,90,92,94,96,97,100,102,104,140,140,144,144,150,151,157,158,162,168,168,173,173]
#comp_init =[fd,fd,fd,fd,fd,fd,fd,fd,fd,fd,fd, 0.00275810142673, 0.00114986769631, 0.0201991683927, 0.002, 0.00645021110853, 0.00243513357583, 0.00857752936147, 0.00154915757877, 0.0103926469702, 0.00241689087521, 0.00707442508677, 0.00103760326605, 0.00711672655612]

#for i in range(0,len(comp_init)):
#    if (Z[i] > 40): 
#        comp_init[i] = comp_init[i]*lf/0.1
#    elif (Z[i] < 20): comp_init[i] = (1-lf)*(1-fd)/3.0
#    else: comp_init[i] = fd*(1 - lf)/8.0


#print(comp_init)
#################################################################################


#Full elements, stealing format from Wollaeger 2021 900 KN Grid, using most common isotopes
#Z= [1,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,46, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 92]
#A= [1,39,40,45,48,51,52,55,56,59,58,63,64,69,74,75,80,79,84,90,106,130,139,140,141,142,145,152,153,158,159,164,165,166,169,174,238]

#Evens 2020 expected Y_e=0.05 composition of dynamical ejecta
#comp_r = [0.,0.,0.,0.,0.,0.,0.,0.,5.32e-6,0.,0.,0.,0.,0.,0.,0.,1.01e-1,2.32e-6,0.,3.72e-1,1.39e-4,3.85e-1,5.11e-4,8.66e-4,8.59e-5,1.5e-3,5.42e-4,2.03e-3,1.55e-3,5.13e-3,3.27e-3,1.40e-2,3.64e-3,1.11e-2,2.34e-3,6.44e-3,8.84e-2]
#comp_no_r = [0.,3.21e-15,1.17e-4,2.39e-10,3.15e-5,1.50e-4,1.27e-1,1.81e-3,8.81e-3,6.53e-4,2.13e-3,6.77e-2,6.28e-2,1.75e-3,4.24e-3,4.34e-4,1.93e-1,2.25e-1,2.90e-1,1.38e-2,5.28e-4,5.56e-7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#comp_CBM = [1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


nx = 80
nz = 2*nx


#volumes = np.zeros((nx,nz))
#for i in range(nx):
#    for j in range(nz):
#        if j == 0:
#            volumes[i,j] = 4*pi*(rZZ[i,j]**2)*np.abs(rmax-rZZ[i,j])
            #print(volumes[i,j])
#        else:
#            if rZZ[i,j] == 0:
#                volumes[i,j] = 4*pi*(rZZ[i,j-1]**2)*np.abs(rZZ[i,j]-rZZ[i,j-1])
#            else:
#                volumes[i,j] = 4*pi*(rZZ[i,j]**2)*np.abs(rZZ[i,j]-rZZ[i,j-1])


def findYe(v_r,pow,Y_d_i,Y_d_o,v_0,angles):
    p1 = 1-(np.abs(np.sin(angles)))**4
    p2 = (((v_r/v_0)**(pow))+1)**-1
    return p1*((Y_d_i-Y_d_o)*p2)+Y_d_o

def findXlanProfile(v_r,pow,X_lan_d_i,X_lan_d_o,theta,v_0):
    angular = (1+((1-np.cos(angles))/(1-np.cos(theta)))**10)**-1
    p1 = 1-(np.abs(np.sin(angles)))**4
    p2 = (((v_r/v_0)**(pow))+1)**-1
    return angular*((X_lan_d_i-X_lan_d_o)*p2)+X_lan_d_o

def findCompfromYe(Ye): #Lippuner 2015 Cutoffs
    if Ye <= 0.19: #produce 3rd peak elements, up to A ~250
        return 3
    if Ye <= 0.1:
        return 2
    if Ye <= 0.15:
        return 1
    else:
        return 0


def make1DSphere(mass=None,v_EK=None,X_lan=None,t=None):
    if None in [mass,v_EK,X_lan,t]:
        print('Model incomplete.')
        return
    print('Velocity is ' + str(v_EK) +'c')
    print('X_lan is '+ str(X_lan))
    print('Mass is ' +str(mass) +' M_sol')
    print('Time is ' +str(t/(24*60*60)) +' days')
    print('1D spherical Model')

    v_t = eta_v*v_EK
    T0 = 10**4 #Kelvin
    rho0 = mass*m_sun/(8*pi*(t**3)*(v_EK*c)**3)
    vmax = 3*v_t*c
    if vmax > c:
        vmax = 2*v_t*c
    print('Maximum velocity is ' +str(vmax/c))
    rmax = texp*vmax
    dr   = rmax/(1.0*nx)
    dv   = vmax/(1.0*nx)
    vx   = np.arange(dv,vmax+0.1,dv)
    rho = np.zeros(nx)
    temp = np.zeros(nx)
    comp = np.zeros((nx,len(Z)))
    erad = np.zeros(nx)
    X_lan_grid = np.zeros(nx)
    X_rproc = np.ones(nx)
    rmin = 0

    #initial temperature calculations
    A1 = 8.4939E+09
    alpha = 1.3642E+00
    eint0 =  A1/(2.0-alpha)*(t/3600.0/24.0)**(1 - alpha)*3600.0*24.0

    if X_lan == 0 and findXlan(Xs) == 0:
        comp_lanth = Xs
    else:
        comp_lanth = changeXlan(Xs,X_lan)
    #print(comp_lanth)
    #comp_lanth = enhanceElement(60,Z,comp_lanth,5)
    print(findXlan(comp_lanth))
    comp_noLanth = changeXlan(Xs,0)
    print(Z)
    print(A)
    print(comp_lanth)
    for i in range(nx):
#        if vx[i] < vmax:
        veff = vx[i]-dv/2
        rho[i] = density_profile(veff,v_t,t,mass,eta_rho,eta_v)
        temp[i] = (eint0*rho[i]/arad)**0.25
        comp[i,:] = comp_lanth
            #Y_e[i] = findYe(veff,0.01,0.5)
#        else:
#            temp[i] = T0*10**-10
#            rho[i] = rho0*10**-40
#            comp[i,:] = comp_noLanth
            #Y_e[i] = 0.5

    fout = h5py.File(name + "_"+"{:.0E}".format(X_lan)+'X_lan_' + str(v_EK)+"v" +"_" + "{:.1E}".format(mass) + 'M' + '_1D.h5','w')
    fout.create_dataset('time',data=[texp],dtype='d')
    fout.create_dataset('Z',data=Z,dtype='i')
    fout.create_dataset('A',data=A,dtype='i')
    fout.create_dataset('rho',data=rho,dtype='d')
    fout.create_dataset('temp',data=temp,dtype='d')
    fout.create_dataset('v',data=vx,dtype='d')
    fout.create_dataset('Xrproc',data=X_rproc,dtype='d')
    fout.create_dataset('comp',data=comp,dtype='d')
    fout.create_dataset('r_out',data=vx*texp,dtype='d')
    fout.create_dataset('r_min',data=[rmin],dtype='d')
    fout.create_dataset('erad',data=erad,dtype='d')




def make2DSphere(mass=None,v_EK=None,X_lan=None,t=None):
    if None in [mass,v_EK,X_lan,t]:
        print('Model incomplete.')
        return
    print('Velocity is ' + str(v_EK) +'c')
    print('X_lan is '+ str(X_lan))
    print('Mass is ' +str(mass) +' M_sol')
    print('Time is ' +str(t/(24*60*60)) +' days')
    print('2D Spherical Model')    

    v_t = eta_v*v_EK
    T0 = 10**4 #Kelvin
    rho0 = mass*m_sun/(8*pi*(t**3)*(v_EK*c)**3)
    #print(changeXlan(X_sol_short,X_lan))
    vmax = 3*v_t*c
    rmax = texp*vmax
    dr   = rmax/(1.0*nx)
    dv   = vmax/(1.0*nx)
    vx   = np.arange(dv,vmax+0.1,dv)
    vz   = np.arange(-1.0*vmax + dv,vmax+0.1,dv)

    rho = np.zeros((nx,nz))
    temp = np.zeros((nx,nz))
    comp = np.zeros((nx,nz,len(Z)))
    erad = np.zeros((nx,nz))
    X_lan_grid = np.zeros((nx,nz))
    X_rproc = np.ones((nx,nz))
    v_r = np.zeros((nx,nz))
    angles = np.zeros((nx,nz))
    vXX = np.repeat(vx[:,np.newaxis],nz,axis=1)
    vZZ = np.reshape(vz,(1,-1))[0]
    vZZ = np.repeat(vZZ[np.newaxis,:],nx,axis=0)
    vRR = (vXX**2+vZZ**2)**0.5
    
    Y_e = findYe(vRR,0.5,0.01,0.5,vx[int(nx/2)],angles)
    
    angles = np.arctan(vZZ/vXX)
    rmin = 0

    rXX = texp*vXX
    rZZ = texp*vZZ
    rRR = texp*vRR


    comp_lanth = changeXlan(X_sol_short,X_lan)
    comp_noLanth = changeXlan(X_sol_short,0)
    print(Z)
    print(A)
    print(changeXlan(X_sol_short,X_lan))

    for i in range(nx):
        for j in range(nz):
            if vRR[i,j] < vmax:
                veff = vRR[i,j]-dv/2
                rho[i,j] = density_profile(veff,v_t,t,mass,eta_rho,eta_v)
                temp[i,j] = T0
                comp[i,j,:] = comp_lanth
                #Y_e[i,j] = findYe(vRR[i,j],0.01,0.5)
            else:
                temp[i,j] = T0*10**-10
                rho[i,j] = rho0*10**-40
                comp[i,j,:] = comp_noLanth
                #Y_e[i,j] = 0.5
               
    fout = h5py.File(name + "_"+"{:.0E}".format(X_lan)+'X_lan_' + str(v_EK)+"v" +"_" + "{:.1E}".format(mass) + 'M' + '_2D.h5','w')
    fout.create_dataset('time',data=[texp],dtype='d')
    fout.create_dataset('Z',data=Z,dtype='i')
    fout.create_dataset('A',data=A,dtype='i')
    fout.create_dataset('rho',data=rho,dtype='d')
    fout.create_dataset('temp',data=temp,dtype='d')
    fout.create_dataset('vx',data=vXX,dtype='d')
    fout.create_dataset('vz',data=vZZ,dtype='d')
    fout.create_dataset('Xrproc',data=X_rproc,dtype='d')
    fout.create_dataset('comp',data=comp,dtype='d')
#    fout.create_dataset('r_out',data=rRR,dtype='d')
    fout.create_dataset('rmin',data=[rmin,-1*rmax],dtype='d')
    fout.create_dataset('erad',data=erad,dtype='d')
    fout.create_dataset('x_out',data=vx*texp,dtype='d')
    fout.create_dataset('z_out',data=vz*texp,dtype='d')                
            
def makePolar(Mp,vp,theta_0,t):
    theta = theta_0*pi/180
    rho_pol = np.zeros((nx,nz))
    angular = (1+((1-np.cos(2*angles))/(1-np.cos(theta)))**10)**-1
    eta_rho = (4*pi*((n_inner-n_outer)/((3-n_inner)*(3-n_outer))))**-1
    eta_v = (((5-n_inner)*(5-n_outer))/((3-n_inner)*(3-n_outer)))**0.5
    v_t = eta_v*vp
    #start off with two broken power laws, apply agnle filter, renormalize to contain full mass? Or multiple eta by 1/factor?
    for i in range(len(vx)):
        for j in range(len(vz)):
            rho_pol[i,j] = density_profile(vRR[i,j],v_t,t,Mp,eta_rho,eta_v)
    
    mtot = sum(rho_pol*volumes*angular)
    mtot = sum(mtot)
    #print(mtot)
    #rho_pol = m_sun*rho_pol*Mp/mtot
    return angular*rho_pol

def density_profile(v_r,v_t,t,M,eta_rho,eta_v):
    if (v_r < v_t*c):
        return (eta_rho*(M*m_sun)*((v_r/(v_t*c))**(-1*n_inner)))/((v_t*c*t)**3)
    else: #(v_r >= v_t)
        return (eta_rho*(M*m_sun)*((v_r/(v_t*c))**(-1*n_outer)))/((v_t*c*t)**3)
        
def makeDyn(Md,vd,ratio,t):
    fun = 1

#colors = findXlanProfile(vRR,4,-2,-6,np.pi/4,vx[int(nx/2)]) #findYe(vRR,0.5,0.01,0.5)

#Multiplication factors to get it to same axes scales as Zappa Paper, t = 20 ms


#plt.scatter(vXX,vZZ,c=colors,cmap='PiYG')
#plt.scatter(-vXX,vZZ,c=colors,cmap='PiYG')
#sm = plt.cm.ScalarMappable(cmap=plt.cm.PiYG,
#                                    norm=plt.Normalize(vmin=min(colors.flatten()),
#                                    vmax=max(colors.flatten())))
#cbar = plt.colorbar(mappable=sm)
#plt.figure(figsize=(10,10))
#plt.xlim(-4.5*10**9,4.5*10**9)
#plt.ylim(-4.5*10**9,4.5*10**9)
#plt.xlabel('Velocity (cm/s)')
#plt.ylabel('Velocity (cm/s)')
#plt.savefig('X_lan.png',format='png')


#Start with model in sphere and then have a Y-e gradient in theta
#make1DSphere(mass=0.04,v_EK=0.1,X_lan=1E-2,t=texp)

#for lan in [1E-9,1E-3,1E-2]: #[1E-9,1E-3,1E-2]:
#    for ms in [0.001,0.003,0.007,0.07,0.1]: #[0.01,0.02,0.03,0.03,0.05]:
#       for v in [0.05,0.1,0.2,0.3]: #[0.05,0.1,0.2,0.3]: #[0.1,0.2,0.3]:
#            make1DSphere(mass=ms,v_EK=v,X_lan=lan,t=texp)






