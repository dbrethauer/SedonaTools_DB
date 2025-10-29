import numpy as np

class element:
    def __init__(self):
        self.lines = {}
        self.upper = {}
        self.lower = {}
        
    def addSeries(self,name,lines,upper,lower):
        self.lines[name] = lines
        self.upper[name] = upper
        self.lower[name] = lower
        
        
H = element()
H.addSeries('Lyman',np.array([121.57,102.57,97.254,94.974,93.780,91.175]),np.array([2,3,4,5,6,np.inf]),1)
H.addSeries('Balmer',np.array([656.3,486.1,434.0,410.2,397.0,364.6]),np.array([3,4,5,6,7,np.inf]),2)
H.addSeries('Paschen',np.array([1875,1282,1094,1005,954.6,820.4]),np.array([4,5,6,7,8,np.inf]),3)
H.addSeries('Brackett',np.array([4051,2625,2166,1944,1817,1458]),np.array([5,6,7,8,9,np.inf]),4)
H.addSeries('Pfund',np.array([7460,4654,3741,3297,3039,2279]),np.array([6,7,8,9,10,np.inf]),5)
H.addSeries('Humphreys',1E3*np.array([12.37,7.503,5.908,5.129,4.673,3.282]),np.array([7,8,9,10,11,np.inf]),6)

    
