# -*- coding: utf-8 -*-
"""
RDKit_Wrapper is a wrapper for RDKit. Its main purpose is to generate initial,
consistent conformers for molecules (consistency maintained using random seeds).
The main function of this module is generateConformer(), which generates a
conformer for a molecule using the MMFF force field implementation of RDKit.

Sections
    . Imports
    
    . Main Functions
        . generateConformer()
        . moleculeFromMol2()
    
    . Auxiliary Functions
        . getInitialConformer()
        . generateCustomMMFF()

Last edit: 2025-01-29
Author: Dinis Abranches, Fathya Salih
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import secrets
import shutil
import copy
import itertools
import warnings

# Specific
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as rdff
from rdkit.Chem import rdFreeSASA

# Local
from lib import VMD_Wrapper as vmd

# =============================================================================
# Main Functions
# =============================================================================

def generateConformer(smilesString,xyzPath=None,calc_energy=False):
    """
    generateConformer() generates an initial, consistent conformer for the
    desired molecule. To do so, it relies on the MMFF force field
    implementation of RDKit (as described in 10.1186/s13321-014-0037-3).
    
    The algorithm was developed with two main objectives:
        1. To consistently return the same conformer for a given molecule;
        2. To minimize the energy of the conformer
    
    The algorithm is divided into three main steps:
        1. Generation of a random initial conformer using the distance geometry
        method implemented in RDKit;
        2. Minimization of the initial conformer with the default
        version of MMFF.
            
    The above steps are repeated thrice and the lowest energy conformation 
    is kept. This redundancy fixes rare cases of bad convergence from the
    intitial structure generated with the distance geometry method.
    
    Parameters
    ----------
    smilesString : string
        SMILES string of the molecule of interest.
    xyzPath : string, optional
        Path where the xyz file of the conformer should be saved. If none, 
        no xyz file is saved.
        The default is None.
    calc_energy : bool, optional
        If True, the function returns the energy and surface area of the
        conformer. If False, only the molecule object is returned.

    Returns
    -------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest with the conformer embedded.

    """
    # Initiate list of energies and list of conformers
    energies=[]
    mList=[]
    for n in range(3): # Generate 3 independent conformers
        # Build molecule object with initial 3D coordinates
        molecule=getInitialConformer(smilesString,randomSeed=n)
        # Skip procedure if number of atoms is inferior to 5
        if molecule.GetNumAtoms()<5:
            if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
            return molecule
        # Generate standard MMFF
        prop,ff=generateCustomMMFF(molecule)    
        # Relax molecule under standard MMFF
        ff.Minimize(10**6)
        # Recenter molecule coordinates
        Chem.rdMolTransforms.CanonicalizeMol(molecule)
        # Append conformer to list of conformers
        mList.append(copy.deepcopy(molecule))
        # Evaluate energy of the conformer and append 
        energies.append(ff.CalcEnergy())
    # Retrieve conformer with the lowest energy
    molecule=mList[np.argmin(energies)]
    # Calculate SASA for selected molecule
    radii=rdFreeSASA.classifyAtoms(molecule)
    sasa=rdFreeSASA.CalcSASA(molecule, radii)
    # If an XYZ file is requested, save XYZ file
    if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
    # Output
    if calc_energy:
        # return molecule,np.max(energies)
        return molecule,energies[np.argmin(energies)],sasa
    else:
        return molecule
   
def moleculeFromMol2(mol2Path, xyzPath=None):
    """
    moleculeFromMol2() generates a molecule object from a mol2 file.
    This allows pre-optimizing a provided geometry using a standard MMFF
    force field.

    Parameters
    ----------
    mol2Path : string
        Path to the mol2 file.
    xyzPath : string, optional
        Path where the xyz file of the conformer should be saved. If none, 
        no xyz file is saved.
        The default is None.

    Returns
    -------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest.

    """
    # Read mol2 file
    molecule=Chem.MolFromMol2File(mol2Path)
    # Add hydrogens to molecule object
    molecule=AllChem.AddHs(molecule)
    # Optimize molecule using standard MMFF
    AllChem.MMFFOptimizeMolecule(molecule)
    # If an XYZ file is requested, save XYZ file
    if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
    # Output
    return molecule

# =============================================================================
# Auxiliary Functions
# =============================================================================

def getInitialConformer(smilesString,randomSeed=42,xyzPath=None):
    """
    !!! Generation seeded !!!
    
    Generate a molecule object from a SMILES string with an initial 3D
    conformer. The initial conformation is obtained using the default DG-based
    algorithm implemented in RDKit and is relaxed using the standard version
    of MMFF.

    Parameters
    ----------
    smilesString : string
        SMILES string of the molecule of interest.

    Returns
    -------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object with an embedded initial conformer.

    """
    # Get molecule object from smiles string
    molecule=Chem.MolFromSmiles(smilesString)
    # Add hydrogens to molecule object
    molecule=AllChem.AddHs(molecule)
    # Generate initial 3D structure of the molecule
    AllChem.EmbedMolecule(molecule,randomSeed=randomSeed)
    # Minimize initial guess with MMFF
    AllChem.MMFFOptimizeMolecule(molecule)
    # If an XYZ file is requested, save XYZ file
    if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
    # Output
    return molecule

def generateCustomMMFF(molecule):
    """
    generateCustomFF() generates custom property and force field objects for
    the inputted molecule. In other words, it performs atom typing and assigns
    MMFF94s parameters to each degree of freedom
    (see 10.1186/s13321-014-0037-3).
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest. Must have already a conformer embedded.

    Returns
    -------
    prop : rdkit.ForceField.rdForceField.MMFFMolProperties
        MMFF molecule property object
    ff : rdkit.ForceField.rdForceField.ForceField object
        Force field object associated to molecule.

    """
    # Initiate props object (rdkit.ForceField.rdForceField.MMFFMolProperties)
    prop=rdff.MMFFGetMoleculeProperties(molecule,
                                                       mmffVariant='MMFF94s')
    ff=rdff.MMFFGetMoleculeForceField(molecule,prop)
    
    # Output
    return prop,ff
    