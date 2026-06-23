"""
Plot the optimization energy profile from an NWChem output file.
Takes the output file path as an argument.

Script structure:
    . Imports
    . Input
    . Functions
    . Plotting

Usage:
    python PlotNwchemEnergies.py <output_file_path>

Last edit: 2026-06-23
Author: Fathya Salih
"""
#!/usr/bin/env python

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

import os.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from lib import NWChem_Wrapper as nwc

#----------------------------------------------
# Input
#----------------------------------------------
# Path to NWChem output file
if len(sys.argv) < 2:
    print("Missing arguments. Usage: PlotNwchemEnergies.py <output_file_path>")
    exit(1)

output_file_path = sys.argv[1]

#----------------------------------------------
# Functions
#----------------------------------------------
def extract_energies(output_file):

    with open(output_file, 'r') as file:
        # read lines
        lines = file.readlines()
        # Check if DFT occurs or not (number of occurences of "Step   0")
        all_starts = nwc.findAllOccurrences(file,['Step', '0'])

        if len(all_starts) < 3:
            no_dft = True
            start_line_hf = all_starts[0]
            start_line_cosmo = all_starts[1]
        else:
            no_dft = False
            start_line_hf = all_starts[0]
            start_line_dft = all_starts[1]
            start_line_cosmo = all_starts[2]

        # define split of energy table headers to find
        first_energy_header = '@ Step Energy Delta E Gmax Grms Xrms Xmax Walltime'.split()
        rest_energy_headers = 'Step Energy Delta E Gmax Grms Xrms Xmax Walltime'.split()

        ## Go to HF section
        # get HF energy lines
        first_hf_en_lines = np.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
        rest_hf_en_lines = np.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
        all_hf_en_lines = np.concatenate((first_hf_en_lines,rest_hf_en_lines))
        # remove lines not in the range [start_hf,start_dft]
        if not no_dft:
            all_hf_en_lines = all_hf_en_lines[all_hf_en_lines < start_line_dft]
        else:
            all_hf_en_lines = all_hf_en_lines[all_hf_en_lines < start_line_cosmo]
        # extract and save energy values
        hf_energies = []; hf_steps = []
        for l in all_hf_en_lines:
            line = lines[l].split()
            if '@' != line[0]:
                step += 1
                E = float(line[1].replace('D', 'E'))
            else:
                step = float(line[1])
                E = float(line[2].replace('D', 'E'))
            hf_energies.append(E)
            hf_steps.append(step)
        # Convert to numpy arrays
        hf_energies = np.array(hf_energies)
        hf_steps = np.array(hf_steps)

        ## Go to DFT section
        if not no_dft:
            # get DFT energy lines
            first_dft_en_lines = np.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
            rest_dft_en_lines = np.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
            all_dft_en_lines = np.concatenate((first_dft_en_lines,rest_dft_en_lines))
            # remove lines not in the range [start_dft,start_cosmo]
            all_dft_en_lines = all_dft_en_lines[all_dft_en_lines < start_line_cosmo]
            all_dft_en_lines = all_dft_en_lines[all_dft_en_lines >= start_line_dft]
            # extract energy values
            dft_energies = []; dft_steps = []
            for l in all_dft_en_lines:
                line = lines[l].split()
                if '@' != line[0]:
                    step += 1
                    E = float(line[1].replace('D', 'E'))
                else:
                    step = float(line[1])
                    E = float(line[2].replace('D', 'E'))
                dft_energies.append(E)
                dft_steps.append(step)
            # Convert to numpy arrays
            dft_energies = np.array(dft_energies)
            dft_steps = np.array(dft_steps)
        else:
            dft_energies = np.zeros_like(hf_energies) * np.nan
            dft_steps = np.zeros_like(hf_steps)

        ## Go to COSMO section
        # get COSMO energy lines
        first_cosmo_en_lines = np.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
        rest_cosmo_en_lines = np.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
        all_cosmo_en_lines = np.concatenate((first_cosmo_en_lines,rest_cosmo_en_lines))
        # remove lines not in the range [start_cosmo,end]
        all_cosmo_en_lines = all_cosmo_en_lines[all_cosmo_en_lines > start_line_cosmo]
        # extract energy values
        cosmo_energies = []; cosmo_steps = []
        for l in all_cosmo_en_lines:
            line = lines[l].split()
            if '@' != line[0]:
                step += 1
                E = float(line[1].replace('D', 'E'))
            else:
                step = float(line[1])
                E = float(line[2].replace('D', 'E'))
            cosmo_energies.append(E)
            cosmo_steps.append(step)
        # Convert to numpy arrays
        cosmo_energies = np.array(cosmo_energies)
        cosmo_steps = np.array(cosmo_steps)
    
    return (hf_steps, hf_energies), (dft_steps, dft_energies), (cosmo_steps, cosmo_energies)

def plot_energies(HF, DFT, COSMO):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    titles = [
        'Energy During HF Optimization',
        'Energy During DFT Optimization',
        'Energy During COSMO Optimization'
        ]
    for i, (steps, energies) in enumerate([HF, DFT, COSMO]):
        axs[i].plot(steps, energies, '-', color='blue')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Energy (a.u.)')
        axs[i].grid(True)
    plt.suptitle(output_file_path)   
    plt.tight_layout()
    plt.show()

def main():
    HF, DFT, COSMO = extract_energies(output_file_path)
    plot_energies(HF, DFT, COSMO)

#----------------------------------------------
# Main
#----------------------------------------------
if __name__ == '__main__':
    main()
