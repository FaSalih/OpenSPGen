#!/usr/bin/env python
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

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

import os.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from lib import spGenerator as spg

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
    HF, DFT, COSMO = spg.extractEnergyProfiles(output_file_path)
    plot_energies(HF, DFT, COSMO)

#----------------------------------------------
# Main
#----------------------------------------------
if __name__ == '__main__':
    main()
