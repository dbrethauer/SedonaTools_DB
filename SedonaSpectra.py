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
from matplotlib.widgets import Slider, Button

c=2.998*10**18 #AA/s
D = 39.5*3.086E24
h = 6.626E-27
erg_to_eV = 6.242E11
distance = D


class SedonaModel:
    def __init__(self,filename=None,gamma=False,polarization=False):
        c = 2.998E18 ##AA/s
        self.name = filename
        self.file = self.name[self.name.rfind('/')+1:]
        self.curves = {}
        #self.angular_indices = {}
        with h5py.File(filename, "r") as f:
            self.freq = np.array(f['nu'])
            self.time = np.array(f['time'])/(60*60*24)
            self.mu   = np.array(f['mu'])
            self.Lnu  = np.array(f['Lnu']).reshape((len(self.time),len(self.freq),len(self.mu)))
            if gamma:
                self.freq *= 1.60218e-6/h
                self.Lnu *= h/1.60218e-6
            self.mu_edges = np.array(f['mu_edges'])
            self.AA   = c/self.freq #AA
            FREQ = np.repeat(np.repeat(self.freq[np.newaxis,:],len(self.time),axis=0)[:,:,np.newaxis],len(self.mu),axis=2)
            self.Llam = FREQ**2*(self.Lnu)/c #erg/s/AA
            if polarization:
                self.Q = np.array(f['Q'])
                self.U = np.array(f['U'])
                self.V = np.array(f['V'])
    def getAngleInd(self,angle=None):
        angle_cos = np.cos(angle*np.pi/180)
        angle_ind = np.searchsorted(self.mu_edges,angle_cos)
        if angle_ind != 0:
            angle_ind -= 1
        return angle_ind
    def getSpec(self,time=None,mode='lam',angle=0,angle_ind=None,symmetric=False):
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
            Y = self.Llam[diff,:,angle_ind]
            Y_s = self.Llam[diff,:,len(self.mu)-1-angle_ind]
        if mode == 'nu':
            Y = self.Lnu[diff,:,angle_ind]
            Y_s = self.Lnu[diff,:,len(self.mu)-1-angle_ind]
            
        if symmetric:
            return (Y+Y_s)/2
        else:
            return Y
            
    def getPolProp(self,prop,time=None,angle=0,angle_ind=None,symmetric=False,nu_min=None,nu_max=None,lam_min=None,lam_max=None):
        if not hasattr(self, prop):
            raise Exception("This spectral set does not have polarization!")
        if nu_min == None and lam_max == None:
            nu_min = self.freq[0]
        elif nu_min == None:
            nu_min = c/lam_max
        if nu_max == None and lam_min == None:
            nu_max = self.freq[len(self.freq)-1]
        elif nu_max == None:
            nu_max = c/lam_min
            
        if angle_ind==None:
            angle_ind = self.getAngleInd(angle=angle)
            
        values = getattr(self, prop)
        
        time = np.array(time)
        timeR = time.reshape(-1,1)
        diff = np.abs(self.time-timeR).argmin(axis=1)
        
        Y = values[diff,(self.freq >= nu_min)&(self.freq<=nu_max),angle_ind]
        Y_s = values[diff,(self.freq >= nu_min)&(self.freq<=nu_max),len(self.mu)-1-angle_ind]
        
        if symmetric:
            Y = (Y+Y_s)/2
        
        return integrate.trapezoid(Y,self.freq)
        
    def getPol(self,time=None,angle=0,angle_ind=None,symmetric=False,nu_min=None,nu_max=None,lam_min=None,lam_max=None):
        if not hasattr(self, 'Q'):
            raise Exception("This spectral set does not have polarization!")
        if nu_min == None and lam_max == None:
            nu_min = self.freq[0]
        elif nu_min == None:
            nu_min = c/lam_max
        if nu_max == None and lam_min == None:
            nu_max = self.freq[len(self.freq)-1]
        elif nu_max == None:
            nu_max = c/lam_min
            
        if angle_ind==None:
            angle_ind = self.getAngleInd(angle=angle)
            
        numeratorQ = self.getPolProp(prop='Q',time=time,angle=angle,angle_ind=angle_ind,symmetric=symmetric,nu_min=nu_min,nu_max=nu_max)
        numeratorU = self.getPolProp(prop='U',time=time,angle=angle,angle_ind=angle_ind,symmetric=symmetric,nu_min=nu_min,nu_max=nu_max)
        numeratorV = self.getPolProp(prop='V',time=time,angle=angle,angle_ind=angle_ind,symmetric=symmetric,nu_min=nu_min,nu_max=nu_max)
        
        denominator = self.getPseudoLum(time=time,nu_max=nu_max,nu_min=nu_min,angle=angle,angle_ind=angle_ind,symmetric=symmetric)
        
        return np.sqrt(numeratorQ**2+numeratorU**2+numeratorV**2)/denominator
        
    def getPolAngle(self,time=None,angle=0,angle_ind=None,symmetric=False,nu_min=None,nu_max=None,lam_min=None,lam_max=None):
        if not hasattr(self, 'Q'):
            raise Exception("This spectral set does not have polarization!")
        if nu_min == None and lam_max == None:
            nu_min = self.freq[0]
        elif nu_min == None:
            nu_min = c/lam_max
        if nu_max == None and lam_min == None:
            nu_max = self.freq[len(self.freq)-1]
        elif nu_max == None:
            nu_max = c/lam_min
            
        if angle_ind==None:
            angle_ind = self.getAngleInd(angle=angle)
            
        numeratorQ = self.getPolProp(prop='Q',time=time,angle=angle,angle_ind=angle_ind,symmetric=symmetric,nu_min=nu_min,nu_max=nu_max)
        numeratorU = self.getPolProp(prop='U',time=time,angle=angle,angle_ind=angle_ind,symmetric=symmetric,nu_min=nu_min,nu_max=nu_max)
        
        return 180+(180/np.pi)*np.arctan(numeratorU/numeratorQ)/2
    
    def getLum(self,time=None,angle=0,angle_ind=None,symmetric=False,nu_min=None,nu_max=None,lam_min=None,lam_max=None):
        if angle_ind==None:
            angle_ind = self.getAngleInd(angle=angle)
        spec = self.getSpec(time=time,mode='nu',angle_ind=angle_ind,symmetric=symmetric)
        return integrate.trapezoid(spec,self.freq)
    def getLumCurve(self,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'Bolometric'+str(costheta) in self.curves.keys():
            lums = self.getLum(self.time,angle_ind=angle_ind,symmetric=symmetric)#np.zeros(len(self.time))
            #for i in range(len(self.time)):
            #    lums[i] = self.getLum(self.time[i],angle_ind=angle_ind)
            self.curves['Bolometric'+str(costheta)] = lums
            return lums
        else:
            return self.curves['Bolometric'+str(costheta)]
    def getPeakLum(self,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'Bolometric'+str(costheta) in self.curves.keys():
            self.getLumCurve(angle_ind=angle_ind,symmetric=symmetric)
            return max(self.curves['Bolometric'+str(costheta)])
        else:
            return max(self.curves['Bolometric'+str(costheta)])
    def getPseudoLum(self,time,nu_max=(2.998E18/3000),nu_min=(2.998E18/25000),angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        spec = self.getSpec(time,'nu',angle_ind=angle_ind,symmetric=symmetric)
        #print(np.shape(spec))
        PseudoSpec = spec[...,(self.freq >= nu_min)&(self.freq<=nu_max)]
        PseudoFreq = self.freq[...,(self.freq >= nu_min)&(self.freq<=nu_max)]
        return integrate.trapezoid(PseudoSpec,PseudoFreq)
    def getPseudoLumCurve(self,nu_max=(2.998E18/3000),nu_min=(2.998E18/25000),angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not 'PseudoBolometric' +str(costheta) in self.curves.keys():
            lums = np.zeros(len(self.time))
            for i in range(len(self.time)):
                lums[i] = self.getPseudoLum(self.time[i],nu_max=nu_max,nu_min=nu_min,angle_ind=angle_ind,symmetric=symmetric)
            self.curves['PsuedoBolometric'+str(costheta)] = lums
            return lums
        else:
            return self.curves['PseudoBolometric'+str(costheta)]
    def getMag(self,time,filt,dist,angle=0,angle_ind=None,symmetric=False): #filter should be in angstrom, use from SVO. SVO has much higher resolution so assumes that
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        
        if not (filt.name+str(costheta) in self.curves.keys()):
            spec = self.getSpec(time,'lam',angle_ind=angle_ind,symmetric=symmetric)/(4*np.pi*dist**2)
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
    def getLightCurve(self,filt,dist=(1E-6*10*3.086E24),angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if not (filt.name+str(costheta) in self.curves.keys()):
            mags = self.getMag(self.time,filt,dist,angle_ind=angle_ind,symmetric=symmetric)#np.zeros(len(self.time))
            #for i in range(len(self.time)):
            #    mags[i] = self.getMag(self.time[i],filt,dist,angle_ind=angle_ind)
            self.curves[filt.name+str(costheta)] = mags - 5*np.log10(dist/(1E-6*10*3.086E24)) #Give distance in cm. Stores absolute magnitudes
            return mags
        else:
            return self.curves[filt.name+str(costheta)] + 5*np.log10(dist/(1E-6*10*3.086E24)) #Returns apparent magnitudes
    def getPeakMag(self,filt,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]
        if filt.name+str(costheta) in self.curves.keys():
            mags = self.curves[filt.name+str(costheta)]
        else:
            tmp = self.getLightCurve(filt,1E-6*10*3.086E24,angle_ind=angle_ind,symmetric=symmetric)
            mags = self.curves[filt.name+str(costheta)]
        peak = min(mags)
        time = self.time[list(mags).index(peak)]
        return time,peak
    def getColorCurve(self,filt1,filt2,angle=0,angle_ind=None,symmetric=False): #returns color1 - color2
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        costheta = self.mu[angle_ind]        
        fakeDist = 1E24
        if not filt1.name+str(costheta) in self.curves.keys():
            mags1 = self.getLightCurve(filt1,fakeDist,angle_ind=angle_ind,symmetric=symmetric) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        else:
            mags1 = self.curves[filt1.name+str(costheta)]
        if not filt2.name+str(costheta) in self.curves.keys():
            mags2 = self.getLightCurve(filt2,fakeDist,angle_ind=angle_ind,symmetric=symmetric) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        else:
            mags2 = self.curves[filt2.name+str(costheta)]
        return mags1-mags2
    def getColor(self,filt1,filt2,time,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)      
        diff = list(np.abs(self.time-time))
        minimum = min(diff)
        ind = diff.index(minimum)
        if minimum > 0.5:
            print('WARNING: NO SPECTRUM TIMES WITHIN 0.5 DAYS OF SELECTED TIME; CHOOSING CLOSEST at t='+str(self.time[ind]) + ' days')
        colors = self.getColorCurve(filt1,filt2,angle_ind=angle_ind,symmetric=symmetric)
        return colors[ind]
    def getRiseRate(self,filt,time,buffer,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)
        fakeDist = 1E24
        currentMags = self.getLightCurve(filt,fakeDist,angle_ind=angle_ind,symmetric=symmetric) - 5*np.log10(fakeDist/(1E-6*10*3.086E24))
        currentBools = ((self.time >= (time-buffer))&(self.time <= (time+buffer)))
            
        currentTimes = self.time[currentBools]
        currentMags = currentMags[currentBools]
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(currentTimes,currentMags)
        
        return slope
    def getTimeAboveHalf(self,filt,dist,angle=0,angle_ind=None,symmetric=False):
        if angle_ind == None:
            angle_ind = self.getAngleInd(angle=angle)       
        time,peak = self.getPeakMag(filt,angle_ind=angle_ind,symmetric=symmetric)
        peak = peak + 5*np.log10(dist/(1E-6*10*3.086E24))
        halfMag = peak-2.5*np.log10(0.5)
        
        mags = self.getLightCurve(filt,dist,angle_ind=angle_ind,symmetric=symmetric)
        
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
                    

                    
                    
    def plotTimeSeriesSpec(self,t_low=1,t_high=5,spacing=1,angle=90,style='-',mode='lam',symmetric=False,log=False):
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                    norm=plt.Normalize(vmin=t_low,
                                    vmax=t_high))

        ts = np.arange(t_low,t_high+spacing,spacing)
        
        if log:
            ts = 10**ts
            colors = sm.to_rgba(np.log10(ts))
        else:
            colors = sm.to_rgba(ts)
        if mode == 'lam':
            Xs = self.AA
            plt.xlabel('Wavelength ($\AA$)',fontsize=14)
            plt.ylabel("Specific Flux (erg/s/cm$^2$/$\AA$)",fontsize=14)
        elif mode == 'nu':
            Xs = self.freq
            plt.xlabel('Frequency (Hz)',fontsize=14)
            plt.ylabel("Specific Flux (erg/s/cm$^2$/Hz)",fontsize=14)

        for q in range(len(ts)):
            plt.errorbar(Xs,self.getSpec(ts[q],mode,angle=angle,symmetric=symmetric)[0],color=colors[q],label="t = " + str(round(ts[q],2)) + " days",linestyle=style)
        plt.xscale('log')
        plt.colorbar(sm,location='right')#,label='Days since Merger')
        y_low, y_high = plt.ylim()
        x_low, x_high = plt.xlim()
        
        if log:
            plt.text(1.25*x_high,10**((np.log10(y_low)+np.log10(y_high))/2),"Time since Merger (d)",ha='center',va='center',rotation='vertical',fontsize=14)
        else:
            plt.text(1.25*x_high,(y_low+y_high)/2,"Time since Merger (d)",ha='center',va='center',rotation='vertical',fontsize=14)

        plt.ylim(y_low,y_high)
        
    def plotAngleSeriesSpec(self,t_obs=5,mode='lam',symmetric=False,style='-'):
    
        if symmetric:
            angle_max = 90
        else:
            angle_max = 180
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow,norm=plt.Normalize(vmin=0,
                                                                   vmax=angle_max))
        if mode == 'lam':
            Xs = self.AA
            plt.xlabel('Wavelength ($\AA$)',fontsize=14)
            plt.ylabel("Specific Flux (erg/s/cm$^2$/$\AA$)",fontsize=14)
        elif mode == 'nu':
            Xs = self.freq
            plt.xlabel('Frequency (Hz)',fontsize=14)
            plt.ylabel("Specific Flux (erg/s/cm$^2$/Hz)",fontsize=14)
            
        
        if symmetric:
            n_mu = np.arange(int(len(self.mu)/2),len(self.mu),1)
        else:
            n_mu = np.arange(0,len(self.mu),1)

        for q in n_mu:
            Y = self.getSpec(time=t_obs,angle=np.arccos(self.mu[q])*180/np.pi,mode=mode,symmetric=symmetric)[0]
            plt.errorbar(Xs,Y,color=sm.to_rgba(np.arccos(self.mu[q])*180/np.pi),label=round(np.arccos(self.mu[q])*180/np.pi,0),linestyle=style)
            
            
    def plotInteractiveSpec(self):
        self.fig, self.ax = plt.subplots(figsize=(12,12))
        plt.subplots_adjust(bottom=0.25)
        
        t_init = 1
        ang_init = 0
        
        self.InteractiveData, = self.ax.plot(self.AA, self.getSpec(time=t_init,angle=ang_init)[0], lw=3, color='blue')
        
        self.ax.set_xlim(np.min(self.AA),np.max(self.AA))
        self.ax.set_xscale('log')
        
        self.ax.set_ylim(0,1.25*np.max(self.getSpec(time=t_init,angle=ang_init)[0]))
        
        self.ax.tick_params(axis='both', which='major', labelsize=24,width=3)
        
        self.symmetricInteractive = False
        self.modeInteractive = 'lam'
        
        self.interactiveMinY = 0
        self.interactiveMaxY = 1.25*np.max(self.getSpec(time=t_init,angle=ang_init)[0])
        self.interactiveFreezeY = False
        
        ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
        ax_angle = plt.axes([0.2, 0.05, 0.65, 0.03])
        self.s_time = Slider(ax_time, 'Time (days)', 0, np.max(self.time), valinit=t_init)
        self.s_angle = Slider(ax_angle, r'cos($\theta$)', -1, 1, valinit=ang_init)
        
        resetax = plt.axes([0.2, 0.15, 0.1, 0.04])
        self.resetButton = Button(resetax, 'Reset', hovercolor='0.975')
        
        modeax = plt.axes([0.4,0.15,0.1,0.04])
        self.modeButton = Button(modeax, 'Lambda', hovercolor='0.975')
        self.modeButton.color='lightblue'
        
        symmax = plt.axes([0.6,0.15,0.1,0.04])
        self.symmButton = Button(symmax, 'Symmetry: OFF', hovercolor='0.975')
        self.symmButton.color = 'red'
        
        yaxmax = plt.axes([0.8,0.15,0.1,0.04])
        self.yaxButton = Button(yaxmax, 'Freeze Y-axis', hovercolor='0.975')
        self.yaxButton.color = 'red'
        
        self.s_time.on_changed(self.updateInteractivePlot)
        self.s_angle.on_changed(self.updateInteractivePlot)
        
        self.resetButton.on_clicked(self.resetInteractive)
        self.symmButton.on_clicked(self.toggleInteractiveSymmetry)
        self.modeButton.on_clicked(self.toggleInteractiveMode)
        self.yaxButton.on_clicked(self.toggleInteractiveFreezeY)
        
    def updateInteractivePlot(self,val):
        t = self.s_time.val
        ang = self.s_angle.val
        
        if self.modeInteractive == 'lam':
            new_x = self.AA
            self.ax.set_xlim(np.min(self.AA),np.max(self.AA))
        else:
            new_x = self.freq
            self.ax.set_xlim(np.min(self.freq),np.max(self.freq))
        
        new_y = self.getSpec(time=t,mode=self.modeInteractive,angle=np.arccos(ang)*180/np.pi,symmetric=self.symmetricInteractive)
        
        if self.interactiveFreezeY:
            self.ax.set_ylim(self.interactiveMinY,self.interactiveMaxY)
        else:
            self.ax.set_ylim(0,1.25*np.max(new_y))
        
        self.InteractiveData.set_ydata(new_y)
        self.InteractiveData.set_xdata(new_x)
        self.fig.canvas.draw_idle()
        
    def resetInteractive(self,event):
        self.s_time.reset()
        self.s_angle.reset()
        
    def toggleInteractiveSymmetry(self,event):
        self.symmetricInteractive = not self.symmetricInteractive
        
        if self.symmetricInteractive:
            self.symmButton.ax.set_facecolor('limegreen')
            self.symmButton.label.set_text('Symmetry: ON')
            self.symmButton.color = 'limegreen'
        else:
            self.symmButton.ax.set_facecolor('red')
            self.symmButton.label.set_text('Symmetry: OFF')
            self.symmButton.color = 'red'
            
        self.updateInteractivePlot(None)
        self.fig.canvas.draw_idle()
        
    def toggleInteractiveMode(self,event):
        if self.modeInteractive == 'lam':
            self.modeInteractive = 'nu'
            #self.modeButton.color = 'lightblue'
            self.modeButton.label.set_text('Freq')
            
        else:
            self.modeInteractive = 'lam'
            #self.modeButton.ax.set_facecolor('lightblue')
            self.modeButton.label.set_text('Lambda')
 
        self.interactiveFreezeY = False
        self.symmButton.ax.set_facecolor('red')
        self.symmButton.label.set_text('Symmetry: OFF')
            
        self.updateInteractivePlot(None)
        self.fig.canvas.draw_idle()
        
    def toggleInteractiveFreezeY(self,event):
        self.interactiveFreezeY = not self.interactiveFreezeY
        
        if self.interactiveFreezeY:
            self.interactiveMinY, self.interactiveMaxY = self.ax.get_ylim()
            self.yaxButton.color = 'limegreen'
        else:
            self.yaxButton.color = 'red'

        self.updateInteractivePlot(None)
        self.fig.canvas.draw_idle()
        
    
        
class TwoComponent(SedonaModel):
    def __init__(self,SM1,SM2):
        self.name1, self.name2 = SM1.name, SM2.name
        if not (SM1.freq == SM2.freq).all() and not (np.shape(SM1.freq) == np.shape(SM2.freq)):
            raise Exception("The frequency grids are not the same")
        if not (SM1.time == SM2.time).all() and not (np.shape(SM1.time) == np.shape(SM2.time)):
            raise Exception("The time grids are not the same")
        if not (SM1.mu_edges == SM2.mu_edges).all() and not (np.shape(SM1.mu_edges) == np.shape(SM2.mu_edges)):
            raise Exception("The angular grids are not the same")
        self.curves = {}
        self.freq = SM1.freq
        self.time = SM1.time
        self.AA = SM1.AA
        self.Lnu = SM1.Lnu + SM2.Lnu
        self.Llam = SM1.Llam + SM2.Llam
        self.mu_edges = SM1.mu_edges
        self.mu = SM1.mu
