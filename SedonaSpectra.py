import numpy as np
import matplotlib.pyplot as plt
import h5py 
import pandas as pd
import json
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, splrep, BSpline
from scipy import stats
import os
import math

c=3*10**18 #AA/s
D = 39.5*3.086E24
h = 6.626E-27
erg_to_eV = 6.242E11
distance = D


class SedonaModel:
    def __init__(self,filename=None):
        c = 3E18 ##AA/s
        self.name = filename
        self.file = self.name[self.name.rfind('/')+1:]
        self.curves = {}
        #self.angular_indices = {}
        with h5py.File(filename, "r") as f:
            self.freq = np.array(f['nu'])
            
            self.time = np.array(f['time'])/(60*60*24)
            self.mu   = np.array(f['mu'])
            self.Lnu  = np.array(f['Lnu']).reshape((len(self.time),len(self.freq),len(self.mu)))
            self.mu_edges = np.array(f['mu_edges'])
            self.AA   = c/self.freq #AA
            FREQ = np.repeat(np.repeat(self.freq[np.newaxis,:],len(self.time),axis=0)[:,:,np.newaxis],len(self.mu),axis=2)
            self.Llam = FREQ**2*(self.Lnu)/c #erg/s/AA
    def getAngleInd(self,angle=None):
        angle_cos = np.cos(angle*np.pi/180)
        angle_ind = np.searchsorted(self.mu_edges,angle_cos)
        if angle_ind != 0:
            angle_ind -= 1
        return angle_ind
    def getSpec(self,time=None,mode='lam',angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        #print(angle_ind)
        time = np.array(time)
        timeR = time.reshape(-1,1)
        diff = np.abs(self.time-timeR).argmin(axis=1)
        #if len(diff) == 1:
        #    diff = diff[0]
        if ((self.time[diff] - time) > 0.5).any():
            print('WARNING: NO SPECTRUM TIMES WITHIN 0.5 DAYS OF SELECTED TIME(S); CHECK TIME INPUT')
        if mode == 'lam':
            return self.Llam[diff,:,angle_ind]
        if mode == 'nu':
            return self.Lnu[diff,:,angle_ind]
    def getLum(self,time=None,angle=0,angle_ind=None):
        if angle_ind==None:
            angle_ind = self.getAngleInd(angle=angle)
        spec = self.getSpec(time=time,mode='nu',angle_ind=angle_ind)
        return integrate.trapezoid(spec,self.freq)
    def getLumCurve(self,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'Bolometric'+str(costheta) in self.curves.keys():
            lums = self.getLum(self.time,angle_ind=angle_ind)#np.zeros(len(self.time))
            #for i in range(len(self.time)):
            #    lums[i] = self.getLum(self.time[i],angle_ind=angle_ind)
            self.curves['Bolometric'+str(costheta)] = lums
            return lums
        else:
            return self.curves['Bolometric'+str(costheta)]
    def getPeakLum(self,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'Bolometric'+str(costheta) in self.curves.keys():
            self.getLumCurve(angle_ind=angle_ind)
            return max(self.curves['Bolometric'+str(costheta)])
        else:
            return max(self.curves['Bolometric'+str(costheta)])
    def getPseudoLum(self,time,nu_max=(3E18/3000),nu_min=(3E18/25000),angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        spec = self.getSpec(time,'nu',angle_ind=angle_ind)
        PseudoSpec = spec[(self.freq >= nu_min)&(self.freq<=nu_max)]
        PseudoFreq = self.freq[(self.freq >= nu_min)&(self.freq<=nu_max)]
        return integrate.trapezoid(PseudoSpec,PseudoFreq)
    def getPseudoLumCurve(self,nu_max=(3E18/3000),nu_min=(3E18/25000),angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'PseudoBolometric' +str(costheta) in self.curves.keys():
            lums = np.zeros(len(self.time))
            for i in range(len(self.time)):
                lums[i] = self.getPseudoLum(self.time[i],nu_max=nu_max,nu_min=nu_min,angle_ind=angle_ind)
            self.curves['PsuedoBolometric'+str(costheta)] = lums
            return lums
        else:
            return self.curves['PseudoBolometric'+str(costheta)]
    def getMag(self,time,filt,dist,angle=0,angle_ind=None): #filter should be in angstrom, use from SVO. SVO has much higher resolution so assumes that
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        
        if not (filt.name+str(costheta) in self.curves.keys()):
            spec = self.getSpec(time,'lam',angle_ind=angle_ind)/(4*np.pi*dist**2)
            #print(max(spec))
            Trans = np.zeros(len(self.AA))#np.where((self.AA >= filt.AA.min())&(self.AA <= filt.AA.max()),,0)#np.zeros(len(self.AA))
            inds = np.searchsorted(filt.AA,self.AA)
            inds = np.where(inds>=len(filt.AA),0,inds)
            Trans = filt.Trans[inds]
            Trans = np.where((self.AA>=filt.AA.min())&(self.AA<=filt.AA.max()),Trans,0)
            filtered = Trans*spec
            if filt.counter == 'energy':
                numerator = integrate.trapezoid(filtered,self.AA) #technically should be negative, but -1/-1 = 1
                denominator = integrate.trapezoid(Trans,self.AA)
                answer = numerator/denominator
                #print(numerator,denominator)
                return np.where(answer!=0,-2.5*np.log10(answer/filt.zeroPoint),45)
                
            elif filt.counter == 'photon':
                numerator = integrate.trapezoid(filtered*self.AA,self.AA) #technically should be negative, but -1/-1 = 1
                denominator = integrate.trapezoid(Trans*self.AA,self.AA)
                answer = numerator/denominator
                #print(numerator,denominator)
                return np.where(answer!=0,-2.5*np.log10(answer/filt.zeroPoint),45)
            else:
                print(filt.counter + " is not a valid counter, please choose energy or photon")
                
        else:
            ind = np.searchsorted(self.time,time)
            return self.curves[filt.name+str(costheta)][ind] + 5*np.log10(dist/(1E-6*10*3.086E24))
    def getLightCurve(self,filt,dist=(1E-6*10*3.086E24),angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not (filt.name+str(costheta) in self.curves.keys()):
            mags = self.getMag(self.time,filt,dist,angle_ind=angle_ind)#np.zeros(len(self.time))
            #for i in range(len(self.time)):
            #    mags[i] = self.getMag(self.time[i],filt,dist,angle_ind=angle_ind)
            self.curves[filt.name+str(costheta)] = mags - 5*np.log10(dist/(1E-6*10*3.086E24)) #Give distance in cm. Stores absolute magnitudes
            return mags
        else:
            return self.curves[filt.name+str(costheta)] + 5*np.log10(dist/(1E-6*10*3.086E24)) #Returns apparent magnitudes
    def getPeakMag(self,filt,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if filt.name+str(costheta) in self.curves.keys():
            mags = self.curves[filt.name+str(costheta)]
        else:
            tmp = self.getLightCurve(filt,1E-6*10*3.086E24,angle_ind=angle_ind)
            mags = self.curves[filt.name+str(costheta)]
        peak = min(mags)
        time = self.time[list(mags).index(peak)]
        return time,peak
    def getColorCurve(self,filt1,filt2,angle=0,angle_ind=None): #returns color1 - color2
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]        
        fakeDist = 1E24
        if not filt1.name+str(costheta) in self.curves.keys():
            mags1 = self.getLightCurve(filt1,fakeDist,angle_ind=angle_ind) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        else:
            mags1 = self.curves[filt1.name+str(costheta)]
        if not filt2.name+str(costheta) in self.curves.keys():
            mags2 = self.getLightCurve(filt2,fakeDist,angle_ind=angle_ind) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        else:
            mags2 = self.curves[filt2.name+str(costheta)]
        return mags1-mags2
    def getColor(self,filt1,filt2,time,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)      
        diff = list(np.abs(self.time-time))
        minimum = min(diff)
        ind = diff.index(minimum)
        if minimum > 0.5:
            print('WARNING: NO SPECTRUM TIMES WITHIN 0.5 DAYS OF SELECTED TIME; CHOOSING CLOSEST at t='+str(self.time[ind]) + ' days')
        colors = self.getColorCurve(filt1,filt2,angle_ind=angle_ind)
        return colors[ind]
    def getRiseRate(self,filt,time,buffer,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        fakeDist = 1E24
        currentMags = self.getLightCurve(filt,fakeDist,angle_ind=angle_ind) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        currentBools = ((self.time >= (time-buffer))&(self.time <= (time+buffer)))
            
        currentTimes = self.time[currentBools]
        currentMags = currentMags[currentBools]
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(currentTimes,currentMags)
        
        return slope
    def getTimeAboveHalf(self,filt,dist,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)       
        time,peak = self.getPeakMag(filt,angle_ind=angle_ind)
        peak = peak + 5*np.log10(dist/(1E-6*10*3.086E24))
        halfMag = peak-2.5*np.log10(0.5)
        
        mags = self.getLightCurve(filt,dist,angle_ind=angle_ind)
        
        above_inds = np.where(mags<=halfMag)
        
        above_times = self.time[above_inds]
        
        return above_times[0],above_times[len(above_times)-1]
    def deleteCurve(self,curve_name):
        if curve_name in self.curves[curve_name]:
            del self.curves[curve_name]
        else:
            print("The curve " + str(curve_name) + " is not in curves")
    def setupSpline(self,filt,dist,n_lim=30,angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)        
        filt_start = min(filt.AA)
        filt_end = max(filt.AA)
        
        n_particles_in_filt = np.zeros(len(self.time))
        with h5py.File(self.name, "r") as f:
            clicks = np.array(f['click'],dtype=int)[angle_ind]
        filt_bools = np.where((self.AA >= filt_start)&(self.AA <= filt_end),1,0).astype(bool)
        #plt.errorbar(self.AA,filt_bools)
        for i in range(len(self.time)):
            n_particles_in_filt[i] = sum(clicks[i][filt_bools])
        bools = np.where(n_particles_in_filt <= n_lim,1,0).astype(bool)
        return bools, n_particles_in_filt

    def fitSpline(self,filt,dist,n_lim=30,setting='low',angle=0,angle_ind=None):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)       
        bools, n_parts = self.setupSpline(filt,dist,n_lim=n_lim,angle_ind=angle_ind)
        is_finite = np.isfinite(self.getLightCurve(filt,dist,angle_ind=angle_ind))
        if setting == 'all':
            bools = np.ones(len(self.time),dtype=bool)
        curve = self.getLightCurve(filt,dist,angle_ind=angle_ind)[is_finite&bools]
        #peakTime,peakMag = self.getPeakMag(filt)
        weights = (1/0.5)*(n_parts[is_finite&bools]/10)**0.5 #reflects that the magnitude error is ~0.75 when there are 10 particles, and the error scales 1/sqrt(N)
        fit = splrep(self.time[is_finite&bools],curve,s=len(curve),w=weights)#*sum(weights)
        return BSpline(*fit)(self.time), bools
    def saveCurves(self,filepath):
        with h5py.File(filepath,'a') as f:
            if 'time' not in list(f.keys()):
                f.create_dataset('time',data=self.time,dtype='f',shape=len(self.time))
            if self.file not in list(f.keys()):
                f.create_group(self.file)
            for q in self.curves.keys():
                if q not in list(f[self.file].keys()):
                    f.create_dataset(self.file+'/'+q,data=self.curves[q],dtype='f',shape=len(self.time))
    def loadCurves(self,filepath):
        with h5py.File(filepath,'r') as f:
            if self.file in list(f.keys()):
                for q in list(f[self.file].keys()):
                    self.curves[q] = np.array(f[self.file+'/'+q])
