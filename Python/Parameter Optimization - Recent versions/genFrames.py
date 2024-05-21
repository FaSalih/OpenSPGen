import sys
sys.path.insert(1,'C:/Users/littl/Box/Research Codes/SPG-main/Python')
from lib import RDKit_Wrapper_13 as rdw
from lib import VMD_Wrapper as vmd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import time

cutoffs = [20]

# for c in range(3):
c=0
cutoff = cutoffs[c]

folder = f'Parameter Optimization - All versions/Parameter Optimization - v20.{c} - HighRes'     
test_mol = pd.read_csv(f'{folder}/Test molecules.csv', header=0)  
    
num_mol = len(test_mol['Name'])

# create directories to store images 
current_dir = os.getcwd()
new_folder_path = os.path.join(current_dir, f"{folder}/Global Optimum")
if not(os.path.exists(new_folder_path)):
    os.mkdir(new_folder_path)
new_folder_path = os.path.join(current_dir, f"{folder}/Local Optimum")
if not(os.path.exists(new_folder_path)):
    os.mkdir(new_folder_path)

# loop over molecules
for mol in range(num_mol):
    mol_name=test_mol['Name'][mol] 
    smilesString=test_mol['SmilesStrings'][mol] 

    optim_param = pd.read_csv(f'{folder}/Optimized (Multi) HBC repulsion parameters.csv', index_col=0).to_numpy()
    global_optimum = optim_param[-1,0:2]
    local_optimum = optim_param[mol,0:2]

    # loop over parameter sets
    for p, param_set in enumerate(['Global', 'Local']):
        if param_set == "Global":
            repulsion_params = global_optimum
        else:
            repulsion_params = local_optimum
            
        repulsion_params = local_optimum
        repulsion_params = global_optimum
        # repulsion_params = np.array([1000.0, 100.0])
        repulsion_params = np.concatenate((repulsion_params, [cutoff]))

        # define path to store xyz files
        if param_set == "Global":
            mol_path=f'{folder}/Global Optimum/{mol_name}'
        else:
            mol_path=f'{folder}/Local Optimum/{mol_name}'
        xyz_path = f'{mol_path}.xyz'

        # generate conformer
        molecule, energies, sasa = rdw.generateConformer(smilesString,repulsion_params,xyz_path,calc_energy=True)

        # generate img from xyz
        img_path = f'{mol_path}.png'
        vmd_path = "C:\\Program Files (x86)\\University of Illinois\\VMD\\vmd.exe"
        vmd.xyz2bmp(xyz_path,img_path,vmd_path)

