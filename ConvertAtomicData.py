import numpy as np
import pandas as pd
import h5py


def makehdf5(root,Z,I,file=None):
    currentRoot = root+f'{Z}/{Z}_{I+1}/'
    #print(currentRoot)
    ion_chi, level_E, level_cs, level_g, level_i, line_A, line_l, line_u, n_levels, n_lines = makeIonGroup(currentRoot)
    file.create_group(ky+'/'+str(ion))
    line_col_id = -1*np.ones(n_lines)
            
            
    file.create_dataset(Z+'/'+str(I)+'/ion_chi',data=ion_chi,dtype='d')
    file.create_dataset(Z+'/'+str(I)+'/n_levels',data=n_levels,dtype='d')
    file.create_dataset(Z+'/'+str(I)+'/n_lines',data=n_lines,dtype='d')
    
    
    file.create_dataset(Z+'/'+str(I)+'/level_g',data=level_g,dtype='uint',shape=(n_levels,))
    file.create_dataset(Z+'/'+str(I)+'/level_E',data=level_E,dtype='d',shape=(n_levels,))
    file.create_dataset(Z+'/'+str(I)+'/level_i',data=level_i,dtype='uint',shape=(n_levels,))
    file.create_dataset(Z+'/'+str(I)+'/level_cs',data=level_cs,dtype=np.int8,shape=(n_levels,))
    file.create_dataset(Z+'/'+str(I)+'/line_col_id',data=line_col_id,dtype=np.int8,shape=(n_lines,))
    file.create_dataset(Z+'/'+str(I)+'/line_A',data=line_A,dtype='d',shape=(n_lines,))
    file.create_dataset(Z+'/'+str(I)+'/line_u',data=line_u,dtype='uint',shape=(n_lines,))
    file.create_dataset(Z+'/'+str(I)+'/line_l',data=line_l,dtype='uint',shape=(n_lines,))
    
def makeIonGroup(currentRoot):
    ion_chi = 0
    level_E = np.array([])
    level_cs = np.array([])
    level_g = np.array([])
    level_i = np.array([])
    line_A = np.array([])
    line_l = np.array([])
    line_u = np.array([])
    n_levels = 0
    n_lines = 0
    
    transitions = getTransitions(currentRoot)
    ion_chi, levels = getEnergyLevels(currentRoot)
    
    level_g = np.array(levels['stat._weight'])
    level_E = np.array(levels['energy[eV]'])
    
    line_u = np.array(transitions['numUlev'])-1
    line_l = np.array(transitions['numLlev'])-1
    line_A = np.array(transitions['G*Einstein'])/level_g[line_u]
    
    n_levels = len(level_E)
    n_lines = len(line_A)
    
    level_cs = -1*np.ones(n_levels)
    level_i = np.arange(0,n_levels,1)
    
    
    return ion_chi, level_E, level_cs, level_g, level_i, line_A, line_l, line_u, n_levels, n_lines


def getTransitions(currentRoot):
    transition = pd.read_csv(currentRoot+'ga.crm',
                    skiprows=9,
                    sep='\s+',#pattern,
                    engine='python',
                    on_bad_lines='skip',
                    index_col=False,
                    header=None,
                    comment='end',
                    names=['numUlev','Ulev','numLlev','Llev','G*Einstein','trans_energy','cStark']
                    #dtype = {'level_name':str,'energy[eV]':np.float64,'stat._weight':np.int32,'average_rad':np.float32,'shell':str,'lastN':np.int8,'parity':str,'highN':np.int8})
                   )
    return transition
    

def getEnergyLevels(currentRoot):
    #file = 'enr.crm'
    level_pattern = r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\{.*?\})\s+(\S+)\s+(\S+)\s+(.*)$'

    table = pd.read_csv(currentRoot+'enr.crm',
                    skiprows=12,
                    sep=level_pattern,
                    engine='python',
                    on_bad_lines='skip',
                    index_col=False,
                    header=None,
                    #comment='end',
                    names=['junk','level_name','energy[eV]','stat._weight','average_rad','shell','lastN','parity','highN'],
                    #dtype = {'level_name':str,'energy[eV]':np.float64,'stat._weight':np.int32,'average_rad':np.float32,'shell':str,'lastN':np.int8,'parity':str,'highN':np.int8})
                   )
    ion_chi = table['junk'][len(table['junk'])-3]
    ion_ind = ion_chi.find('IONPOT')
    ion_ind_0 = ion_chi.find('NEXTGROUND')
    ion_chi = float(ion_chi[ion_ind+len('IONPOT='):ion_ind_0])

    table = table[~table['level_name'].isnull()]
    table = table[['level_name','energy[eV]','stat._weight']]
    
    return ion_chi, table
    
    
    
new_name = './Banerjee_atomic_data.h5'

approvedAtoms = ['57']

with h5py.File(new_name,'w') as newf:
    newf.create_dataset('file_version',data=1)
    for ky in approvedAtoms:
        for ion in [4,8]:#[0,1,2,3,4,5,6,7,8,9,10]:
            makehdf5('./TestAtomicData/',Z=ky,I=ion,file=newf)
