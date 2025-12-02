import os
import numpy as np

def setup(path,cleanup=False,coreSpec=False):
    files = os.listdir(path)
    if os.path.exists('./FinalSpectra') == False:
        os.mkdir('./FinalSpectra')
    for file in files:
        if file[-2:].strip()=='h5':
            #print(file)
            if os.path.exists("./FinalSpectra/"+file[:-3]+"_spec_final.h5"):
                print('Already completed spectra for file: ' + file + '; moving on.')
            else:
                if os.path.exists("./"+file[:-3]) == False:
                    os.mkdir("./"+file[:-3])
                os.system("cp "+path + file + " " + "./"+file[:-3]+"/model.h5")
                if 'Magnetar' or 'Accretion' in file:
                    changeLua(file)
                else:
                    os.system("cp ./kilonova.lua " + "./"+file[:-3]+"/kilonova.lua")
                if coreSpec:
                    os.system("cp "+path + file + "_corespec.txt " + "./"+file[:-3]+"/corespec.txt")
                runSedona(file)
                if cleanup == True:
                    os.system("rm ./"+file[:-3]+"/model_spec*")
                    os.system("rm ./"+file[:-3]+"/plt*")

def runSedona(file):
   os.system("cd ./" + file[:-3] + "; sedona6.ex kilonova.lua; cd ./..")
   os.system("mv ./" + file[:-3] + "/model_spec_final.h5 ./FinalSpectra/"+file[:-3]+"_spec_final.h5")
   
def changeLua(file):
    mode = None
    if 'Magnetar' in file:
        E_rot = file.find('Erot')
        E_rot = float(file[E_rot-7:E_rot])
        t_0 = file.find('t0_')
        t_0 = float(file[t_0-7:t_0])
        #print(E_rot, t_0)
        mode = 'Magnetar'
    elif 'Accretion' in file:
        M_dot = 1
        mode = 'Accretion'
    newFunction = ""
    if mode == 'Magnetar':
        newFunction = "t_0 = " + str(t_0*24*60*60) + "\nE_0 = " + str(E_rot) + "\nL_0 = E_0/t_0\nfunction core_luminosity(t)\n\tL = L_0/(1+t/t_0)/(1+t/t_0)\n\treturn L\nend\n"
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
        

setup("/home/dbrethauer/kn_project/grid/grid_practice/models/",cleanup=False)
#changeLua('Magnetar_1.0E+49Erot_5.0E-01t0_3E-2M_0.1v_1D.h5')

