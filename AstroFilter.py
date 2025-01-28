class AstroFilter:
    def __init__(self,filename,mag_0,center,counter='photon'):
        self.full = np.genfromtxt('./../Filters/'+filename, names=['AA','transmission'])
        self.AA = self.full['AA']
        self.Trans = self.full['transmission']
        self.zeroPoint = mag_0 #units of erg/s/cm^2/AA
        name_start = filename.find('_')+1
        self.name = filename[name_start:-4]
        self.wavecenter = center #units of AA
        self.counter = counter.lower()
