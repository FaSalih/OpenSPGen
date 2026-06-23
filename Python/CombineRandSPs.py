#!/usr/bin/env python
"""
Combine SPs calculated from different random conformers into averaged SPs.
Allows using an arithmetic averaging scheme (default) or a Boltzmann
averaging scheme (requires presence of output.nw or outputSummary.nw files).
If certain job (conformer) files are missing, the SP is replaced with an nan
vector of appropriate dimensions and excluded from the averaging.

Script structure:
    . Imports
    . Input Parsing
    . Dataset Combination Loop

Usage:
    python CombineRandSPs.py <number_of_confs> 

Last edit: 2026-06-23
Author: Fathya Salih
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os, sys, glob
import numpy as np
import pandas as pd
from tqdm import tqdm 

# Local
import os.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from lib import spGenerator as spg

# =============================================================================
# Configuration
# =============================================================================
if len(sys.argv) < 2:
    print("Error: Missing arguments. Usage: CombineRandSPs.py <number_of_confs>")
    exit(1)
else:
    try:
        int(sys.argv[1].isnumeric())
    except ValueError:
        print(f"Error: Incorrect argument type. The provided number of conformers should be a an integer. Received: {sys.argv[1]}")
        exit(1)

# define number of conformer jobs to average over and the averaging scheme
n_confs = int(sys.argv[1])
avg_scheme = 'arithmetic' # allows: 'arithmetic' or 'boltzmann'
avg_scheme = 'boltzmann' # allows: 'arithmetic' or 'boltzmann'

# get all job folders existing in working directory
pattern = os.path.join('.', "SP-RandInitXYZ-Mol*")
all_dirs = glob.glob(pattern)

# read a sample folder to get charge range (array sizes)
sample_dir = f'{all_dirs[0]}/{all_dirs[0]}_0'
sample_SP = pd.read_csv(f'{sample_dir}/sigmaProfile.csv', header=None).to_numpy().astype(float)
sigma = sample_SP[:,0]
n_bins = len(sigma)
n_mols = len(all_dirs)

# initialize arrays for averaged SPs
mean_SPs = np.zeros((n_mols, n_bins))
std_SPs = np.zeros((n_mols, n_bins))
mol_names = [] # could be defined separately, if molecule names and indices are different 

# loop over molecules
for m, mol_dir in enumerate(tqdm(all_dirs)):
    # get molecule name/identifiers
    mol_name = mol_dir.split('/')[-1].split('_')[-1]
    mol_names.append(mol_name)

    # initialize array for conformer SPs for current molecule
    mol_sps = np.zeros((n_confs, n_bins))
    cosmo_en = np.zeros((n_confs, 1))

    # loop over conformer jobs
    for job in range(n_confs):
        job_folder = f"{mol_dir}/{mol_dir}_{job}"
        # read csv in molecule folder
        try:
            SP = pd.read_csv(f'{job_folder}/sigmaProfile.csv', header=None).to_numpy().astype(float)[:,1]
            if avg_scheme == 'boltzmann': 
                try:
                    _, _, cosmo_energies = spg.extractEnergyProfiles(f'{job_folder}/output.nw')
                    cosmo_energy = cosmo_energies[1][-1]
                except FileNotFoundError:
                    try: cosmo_energy = spg.extractFinalEnergy(f'{job_folder}/outputSummary.nw')
                    except FileNotFoundError:
                        print(f'No suitable output files were found to extract conformer energy for boltzmann-averaging. Skipping...')
                    cosmo_energy = np.nan
            else:
                cosmo_energy = np.nan

        except FileNotFoundError:
            print(f'sigmaProfile.csv file not found in {job_folder}. Skipping...')
            SP = np.ones((n_bins)) * np.nan
            cosmo_energy = np.nan

        # save to molecule SP array
        mol_sps[job,:] = SP
        # save to energies array
        cosmo_en[job] = float(cosmo_energy) *  2625.5     # convert hartree to [kJ/mol]

    # save averaged sigma profile to array
    if avg_scheme == 'arithmetic': 
        mean_SPs[m,:] = np.nanmean(mol_sps, axis=0)
        std_SPs[m,:] = np.nanstd(mol_sps, axis=0)

    if avg_scheme == 'boltzmann': 
        # Convert raw energies to deltaE
        dE = cosmo_en - np.min(cosmo_en)
        # get boltzmann factors
        kT = 0.0083144622 * 300     # (boltzmann constant * Temp) [kJ/mol]
        boltz_wts = np.exp(-dE/kT)
        # calculate and save boltzmann averaged sigma profiles
        ba_mean = np.sum(mol_sps*boltz_wts, axis=0)/np.sum(boltz_wts)
        ba_var = np.sum((mol_sps*boltz_wts - ba_mean)**2, axis=0)/np.sum(boltz_wts)
        mean_SPs[m,:] = ba_mean
        std_SPs[m,:] = np.sqrt(ba_var)

# create dataframes
means_df = pd.DataFrame(mean_SPs, columns=sigma, index=mol_names)
std_df = pd.DataFrame(std_SPs, columns=sigma, index=mol_names)

means_df.index.name = 'Index'
std_df.index.name = 'Index'

# save to csvs
if avg_scheme == 'arithmetic': 
    means_path = f"sp_gen_{n_confs}SPs.csv"
    stds_path = f"sp_gen_{n_confs}SPs_std.csv"
    
if avg_scheme == 'boltzmann': 
    means_path = f"sp_gen_ba_{n_confs}SPs.csv"
    stds_path = f"sp_gen_ba_{n_confs}SPs_std.csv"

means_df.to_csv(means_path, index=True)
std_df.to_csv(stds_path, index=True)


print(f'Complete!\nSaved {avg_scheme}-averaged SPs to {means_path} and their standard deviations to {stds_path}')