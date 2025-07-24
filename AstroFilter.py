import numpy as np
import matplotlib.pyplot as plt

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
        
        
LSSTu = AstroFilter('LSST_LSST.u.dat',8.03787e-9,3694.25)
LSSTg = AstroFilter('LSST_LSST.g.dat',4.7597e-9,4840.83)
LSSTr = AstroFilter('LSST_LSST.r.dat',2.8156e-9,6257.74)
LSSTi = AstroFilter('LSST_LSST.i.dat',1.91864e-9,7560.48)
LSSTz = AstroFilter('LSST_LSST.z.dat',1.44312e-9,8701.29)
LSSTy = AstroFilter('LSST_LSST.y.dat',1.14978e-9,9749.3)

LSST = [LSSTu,LSSTg,LSSTr,LSSTi,LSSTz,LSSTy]

##

JWSTF560 = AstroFilter('JWST_MIRI.F560W.dat',3.42783e-11,56651.28)
JWSTF770 = AstroFilter('JWST_MIRI.F770W.dat',1.86525e-11,77111.48)
JWSTF1000 = AstroFilter('JWST_MIRI.F1000W.dat',1.09883e-11,99981.09)
JWSTF1130 = AstroFilter('JWST_MIRI.F1130W.dat',8.5121e-12,113159.44)
JWSTF1280 = AstroFilter('JWST_MIRI.F1280W.dat',6.63345e-12,128738.34)
JWSTF1500 = AstroFilter('JWST_MIRI.F1500W.dat',4.79728e-12,151469.07)
JWSTF1800 = AstroFilter('JWST_MIRI.F1800W.dat',3.3658e-12,180508.30)
JWSTF2100 = AstroFilter('JWST_MIRI.F2100W.dat',2.51726e-12,209373.19)
JWSTF2550 = AstroFilter('JWST_MIRI.F2550W.dat',1.69204e-12,254994.20)

MIRI = [JWSTF560,JWSTF770,JWSTF1000,JWSTF1130,JWSTF1280,JWSTF1500,JWSTF1800,JWSTF2100,JWSTF2550]

JWSTF070 = AstroFilter('JWST_NIRCam.F070W.dat',2.1969e-9,7088.30)
JWSTF090 = AstroFilter('JWST_NIRCam.F090W.dat',1.33748e-9,9083.40)
JWSTF115 = AstroFilter('JWST_NIRCam.F115W.dat',8.17032e-10,11623.88)
JWSTF150 = AstroFilter('JWST_NIRCam.F150W.dat',4.83319e-10,15104.23)
JWSTF200 = AstroFilter('JWST_NIRCam.F200W.dat',2.75252e-10,20028.15)
JWSTF277 = AstroFilter('JWST_NIRCam.F277W.dat',1.42719e-10,27844.64)
JWSTF356 = AstroFilter('JWST_NIRCam.F356W.dat',8.54888e-11,35934.49)
JWSTF444 = AstroFilter('JWST_NIRCam.F444W.dat',5.61165e-11,44393.52)

NIRCam = [JWSTF070,JWSTF090,JWSTF115,JWSTF150,JWSTF200,JWSTF277,JWSTF356,JWSTF444]

##

MASSJ = AstroFilter('2MASS_2MASS.J.dat',7.12762e-10,12350,counter='energy')
MASSH = AstroFilter('2MASS_2MASS.H.dat',1.11933e-10,16620,counter='energy')
MASSKs =AstroFilter('2MASS_2MASS.Ks.dat',4.20615e-11,21590,counter='energy')

MASS = [MASSJ,MASSH,MASSKs]

sm = plt.cm.ScalarMappable(cmap=plt.cm.copper,
                                    norm=plt.Normalize(vmin=0,
                                    vmax=len(NIRCam)))

colors = {'u':'m','g':'g','r':'r','i':'lightcoral','z':'brown','y':'maroon','J':'orange','H':'darkgoldenrod','Ks':'darkslategray'}
colors['070'] = sm.to_rgba(0)
colors['090'] = sm.to_rgba(1)
colors['115'] = sm.to_rgba(2)
colors['150'] = sm.to_rgba(3)
colors['277'] = sm.to_rgba(4)
colors['356'] = sm.to_rgba(5)
colors['444'] = sm.to_rgba(6)

sm = plt.cm.ScalarMappable(cmap=plt.cm.winter_r,
                                    norm=plt.Normalize(vmin=0,
                                    vmax=len(MIRI)))

colors['560'] = sm.to_rgba(0)
colors['770'] = sm.to_rgba(1)
colors['1000'] = sm.to_rgba(2)
colors['1130'] = sm.to_rgba(3)
colors['1280'] = sm.to_rgba(4)
colors['1500'] = sm.to_rgba(5)
colors['1800'] = sm.to_rgba(6)
colors['2100'] = sm.to_rgba(7)
colors['2550'] = sm.to_rgba(8)
to_model = {'u':LSSTu,'g':LSSTg,
            'r':LSSTr,'i':LSSTi,
            'z':LSSTz,'y':LSSTy,
            'J':MASSJ,'H':MASSH,
            'Ks':MASSKs,'070':JWSTF070,
            '090':JWSTF090,'115':JWSTF115,
            '150':JWSTF150,'277':JWSTF277,
            '356':JWSTF356,'444':JWSTF444,
            '560':JWSTF560,'770':JWSTF770,
            '1000':JWSTF1000,'1130':JWSTF1130,
            '1280':JWSTF1280,'1500':JWSTF1500,
            '1800':JWSTF1800,'2100':JWSTF2100,
            '2550':JWSTF2550}
