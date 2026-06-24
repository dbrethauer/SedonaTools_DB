import os
import shutil
import subprocess
import numpy as np

def setup(path,cleanup=False,coreSpec=False):
    if not path.endswith('/'):
        path += '/'
    files = os.listdir(path)
    if not os.path.exists('./FinalSpectra'):
        os.mkdir('./FinalSpectra')
    for file in files:
        if file.strip().endswith('h5'):
            #print(file)
            name = file[:-3]
            model_dir = './'+name
            if os.path.exists("./FinalSpectra/"+name+"_spec_final.h5"):
                print('Already completed spectra for file: ' + file + '; moving on.')
                continue
            else:
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                shutil.copy(os.path.join(path,file),os.path.join(model_dir, "model.h5"))
                
                if 'Magnetar' in file or 'Accretion' in file:
                    changeLua(file)
                else:
                    shutil.copy("./kilonova.lua", os.path.join(model_dir,'kilonova.lua'))
                if coreSpec:
                    shutil.copy(f'{path}{name}_corespec.txt',os.path.join(model_dir,'corespec.txt'))
                runSedona(file)
                if cleanup == True:
                    os.system("rm ./"+name+"/model_spec*")
                    os.system("rm ./"+name+"/plt*")

def runSedona(file):
    command = ["mpiexec","sedona6.ex","kilonova.lua"]
    model_dir = './'+file[:-3]
    environment = os.environ.copy()
    try:
        subprocess.run(command,cwd=model_dir,env=environment,check=True)
        shutil.move(os.path.join(model_dir,'model_spec_final.h5'),os.path.join('FinalSpectra',file[:-3]+'_spec_final.h5'))
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: Sedona failed on {file} with exit code {e.returncode}")
        
    
   
def changeLua(file):
    mode = None
    if 'Magnetar' in file:
        L_0 = file.find('L0')
        L_0 = float(file[L_0-7:L_0])
        t_0 = file.find('t0_')
        t_0 = float(file[t_0-7:t_0])
        #print(E_rot, t_0)
        mode = 'Magnetar'
    elif 'Accretion' in file:
        M_dot = 1
        mode = 'Accretion'
    newFunction = ""
    if mode == 'Magnetar':
        newFunction = "t_0 = " + str(t_0*24*60*60) + "\nL_0 = " + str(L_0) + "\nfunction core_luminosity(t)\n\tL = L_0/(1+t/t_0)/(1+t/t_0)\n\treturn L\nend\n"
    with open("kilonova.lua", "r") as fin, open("./"+file[:-3]+"/kilonova.lua", "w") as fout:
        fullText = fin.read()
        ind_core = fullText.find('core_luminosity')
        if ind_core == -1:
            newText = fullText + "\n" + newFunction
            
        else:
            funcBool = fullText.find('function core_luminosity')
            if funcBool == -1:
                lineEnd = fullText.find("\n",ind_core)
                oldLine = fullText[ind_core:lineEnd]
            
            else:
                funcEnd = fullText.find('end', funcBool)
                oldLine = fullText[funcBool:funcEnd+len('end')]
 
            newText = fullText.replace(oldLine,newFunction)
        fout.write(newText)
        

#setup("/home/dbrethauer/kn_project/grid/grid_practice/models/",cleanup=False)
#changeLua('Magnetar_1.0E+49Erot_5.0E-01t0_3E-2M_0.1v_1D.h5')
