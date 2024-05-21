# -*- coding: utf-8 -*-
"""
RDKit_Wrapper is a wrapper for RDKit. Its main purpose is to generate initial,
consistent conformers for molecules (i.e., conformers with maximal surface area
and minimal intramolecular interactions). The main function, 
generateConformer(), contains the essential details of the algorithm.

Copied from RDKit_Wrapper_13.py

Sections
    . Imports
    
    . Main Functions
        . generateConformer()
        . testConsistency()
    
    . Auxiliary Functions
        . getInitialConformer()
        . generateFF()
        . fixBonds()
        . fixAngles()
        . fixDoubleBondDihedrals()
        . fixOHdihedrals()
        . addRepulsion()
        . isHB()
        . getNeighborsTree()

Last edit: 2024-05-20
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
# from tqdm import tqdm
# import imageio
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as rdff
from rdkit.Chem import rdFreeSASA

# Local
from lib import VMD_Wrapper as vmd

# =============================================================================
# Main Functions
# =============================================================================

def generateConformer(smilesString,repulsion_params=None,xyzPath=None,calc_energy=False):
    """
    generateConformer() generates an initial, consistent conformer for the
    desired molecule. To do so, it relies on the MMFF force field
    implementation of RDKit (as described in 10.1186/s13321-014-0037-3).
    
    The algorithm was developed with two main objectives:
        1. To consistently return the same conformer for a given molecule;
        2. To maximize the available surface area of the conformer and minimize
        its intramolecular interactions.
    
    The algorithm is divided into three main steps:
        1. Generation of an initial conformer using the distance geometry
        method implemented in RDKit;
        2. Minimization of the initial conformer with a customized version of
        MMFF, namely including fixed bonds, fixed angles, fixed double bond
        dihedrals, fixed OH dihedrals (some cases) and all-atom repulsion;
        3. Relaxation of the structure obtained in step 2 with the default
        version of MMFF (maintaining some OH dihedrals).
    
    The above steps are repeated thrice and any outlier conformation is
    discarded. This redundancy fixes rare cases of bad convergence from the
    intitial structure generated with the distance geometry method.
    
    The custom MMFF has the following energy hierarchy (R signifies an atom
    that is neither an HBD or HBA):
        1. Out-of-plane bending
        2. Fixed double bond dihedrals
        3. Fixed angles
        4. HBD-HBA repulsion
        5. HBD/HBA-R repulsion
        6. R-R repulsion

    Parameters
    ----------
    smilesString : string
        SMILES string of the molecule of interest.
    xyzPath : string, optional
        Path where the xyz file of the conformer should be saved. If none, 
        no xyz file is saved.
        The default is None.

    Returns
    -------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest with the conformer embedded.

    """
    # Extract or set repulsion parameters
    if repulsion_params is None:
        # Generate molecule object from smiles string
        molecule=getInitialConformer(smilesString)
        # Check if molecule is capable of forming internal hydrogen bonds
        if internalHBs(molecule):
            repulsion_params=[215.44,9.33,20] # optimized from optimParam_v19 - Multi param set
        else:
            repulsion_params=[1.0,1.0,20]
    # Initiate list of energies and list of conformers
    energies=[]
    mList=[]
    for n in range(3): # Generate 3 independent conformers
        # Build molecule object with initial 3D coordinates
        molecule=getInitialConformer(smilesString)
        # Skip procedure if number of atoms is inferior to 5
        if molecule.GetNumAtoms()<5:
            if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
            return molecule
        # Generate custom force fields
        prop,ff=generateCustomMMFF(molecule,2)
        # Fix double bond dihedrals
        ff=fixDoubleBondDihedrals(molecule,prop,ff,10**1)
        # Add artificial repulsion
        ff=addRepulsion(molecule,ff,10**-3,repulsion_params)
        # Geometry optimization with custom force field
        ff.Minimize(10**6)
        # Generate standard MMFF
        prop,ff=generateCustomMMFF(molecule,3)    
        # Relax molecule under standard MMFF
        ff.Minimize(10**6)
        # Recenter molecule coordinates
        Chem.rdMolTransforms.CanonicalizeMol(molecule)
        # Append conformer to list of conformers
        mList.append(copy.deepcopy(molecule))
        # Evaluate energy of the conformer and append 
        energies.append(ff.CalcEnergy())
    # Calculate the energy difference between the 5 conformers generated
    eDiff=[]
    energies=np.array(energies)
    for n in range(len(mList)): 
        eDiff.append(np.abs(energies[n] - energies).sum())
    # Retrieve conformer with the smallest relative energy difference
    molecule=mList[eDiff.index(min(eDiff))]
    # Calculate SASA for selected molecule
    radii=rdFreeSASA.classifyAtoms(molecule)
    sasa=rdFreeSASA.CalcSASA(molecule, radii)
    # If an XYZ file is requested, save XYZ file
    if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
    # Output
    if calc_energy:
        # return molecule,np.max(energies)
        return molecule,energies[eDiff.index(min(eDiff))],sasa
    else:
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
    # Minimizie initial guess with MMFF
    AllChem.MMFFOptimizeMolecule(molecule)
    # If an XYZ file is requested, save XYZ file
    if xyzPath is not None: Chem.MolToXYZFile(molecule,xyzPath)
    # Output
    return molecule

def generateCustomMMFF(molecule,variant):
    """
    generateCustomFF() generates custom property and force field objects for
    the inputted molecule. In other words, it performs atom typing and assigns
    MMFF94s parameters to each degree of freedom
    (see 10.1186/s13321-014-0037-3).
    Variant 1 is the standard MMFF94s. Variant 2 is the custom MMFF employed in
    this work, where the following energy contributions are switched off:
        . VdW
        . Ele
        . StretchBend
        . Torsion
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest. Must have already a conformer embedded.
    variant : int
        Variant of the custom force field. One of:
            1 - Regular MMFF94s
            2 - Custom MMFF94s

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

    if variant==2:
        # Turn off specific contributions (see 10.1186/s13321-014-0037-3)
        prop.SetMMFFVdWTerm(False) # Turn off van der Waals interactions
        prop.SetMMFFEleTerm(False) # Turn off electrostatic interactions
        prop.SetMMFFStretchBendTerm(False) # Turn off stretch-bend
        prop.SetMMFFTorsionTerm(False) # Turn off torsional
        # Generate force field object
        ff=rdff.MMFFGetMoleculeForceField(molecule,prop,
                                                        nonBondedThresh=20,
                                                ignoreInterfragInteractions=False)
    
    # Output
    return prop,ff

def fixDoubleBondDihedrals(molecule,prop,ff,force):
    """
    fixDoubleBondDihedrals() adds dihedral constraints to the force field
    object ff, such that the dihedral amplitude of each double bond (including
    bonds with order=1.5) in the molecule does not change upon optimization.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest. Must have already a conformer embedded.
    prop : rdkit.ForceField.rdForceField.MMFFMolProperties
        MMFF molecule property object of interest.
    ff : rdkit.ForceField.rdForceField.ForceField object
        Force field object of interest.
    force : int, optional
        Force constant to be applied to each dihedral when they deviate from
        the requested amplitude (as defined in 10.1186/s13321-014-0037-3).

    Returns
    -------
    ff : rdkit.ForceField.rdForceField.ForceField object
        Force field object as inputted but with added dihedral constraints.

    """
    # Iterate over all bonds in the molecule
    for bond in molecule.GetBonds():
        # Get bond order
        order=bond.GetBondTypeAsDouble()
        # Check if this is a double bond
        if order==1.5 or order==2.0:
            # Assign the indexes of the double bonded atoms as dihedral centers
            atom2=bond.GetBeginAtomIdx()
            atom3=bond.GetEndAtomIdx()
            # Get neighbors of atom2 and atom3
            neighList2=molecule.GetAtomWithIdx(atom2).GetNeighbors()
            neighList3=molecule.GetAtomWithIdx(atom3).GetNeighbors()
            # Convert rdkit atom objects to index numbers
            neighList2=[neigh.GetIdx() for neigh in neighList2]
            neighList3=[neigh.GetIdx() for neigh in neighList3]
            # Remove atom2 and atom3 from neighLists
            neighList2.remove(atom3)
            neighList3.remove(atom2)
            # If a neighbor list is empty, double bond is terminal; skip
            if not neighList2 or not neighList3: continue
            # Add torsion constraints to all diherals
            for atom1 in neighList2:
                for atom4 in neighList3:
                    ff.MMFFAddTorsionConstraint(atom1,
                                                atom2,
                                                atom3,
                                                atom4,
        True, # Relative to current dihedral, should be changed to round values
                                                0, # Min deviation
                                                0, # Max deviation
                                                force)
    # Output
    return ff

def addRepulsion(molecule,ff,force,params):
    """
    addRepulsion() adds artificial all-atom repulsion to the force field.
    This is achieved by defining artificial distance constraints. The repulsion
    is felt within a 10 angstrom radius of each atom. No repulsion is added
    between atoms that are 1-2 or 1-3 neighbors. Three types of repulsion are
    used, ordered from strongest to weakest (R signifies an atom that is
    neither an HBD or HBA):
          HBD-HBA: repulsion with strength 10*force; not added for atoms that
                   are 1-4 neighbors (10 anstrom radius).
        HBD/HBA-R: repulsion with strength 2*force (10 angstrom radius).
              R-R: repulsion with strength force (10 angstrom radius).

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest. Must have already a conformer embedded.
    ff : rdkit.ForceField.rdForceField.ForceField object
        Force field object of interest. Not to be confused with an
        rdkit.ForceField.rdForceField.MMFFMolProperties object.
    force : int, optional
        Force constant to be applied to each atom pair (distance constraint,
        as defined in 10.1186/s13321-014-0037-3).
        Cannot be too large, otherwise it becomes stronger than the forces set
        to fix bond length and angle amplitude (i.e., molecule breaks apart).

    Returns
    -------
    ff : rdkit.ForceField.rdForceField.ForceField object
        Force field object as inputted but with artificial, all-atom repulsion.

    """
    HBD_HBA_f,HBC_R_f,cutoff=params
    R_R_f=1
    # Double loop iterating over all atoms of the molecule
    for i in range(molecule.GetNumAtoms()-1):
        for j in range(i+1,molecule.GetNumAtoms()):
            # If atoms are 1-2 or 1-3 or 1-4 neighbors, skip iteration
            if i not in getNeighborsTree(molecule,j,4):
                # Condition for atoms to hydrogen bond
                cond1=((isHB(molecule,i)=='HBD' and isHB(molecule,j)=='HBA') 
                       or
                       (isHB(molecule,i)=='HBA' and isHB(molecule,j)=='HBD'))
                # Condition for HBD/HBA-R repulsion
                cond3=(isHB(molecule,i)=='HBD'
                      or isHB(molecule,i)=='HBA'
                      or isHB(molecule,j)=='HBD' 
                      or isHB(molecule,j)=='HBA')
                # HBD-HBA Repulsion
                if cond1:
                    # Add distance constraint (10.1186/s13321-014-0037-3)
                    ff.MMFFAddDistanceConstraint(i, # First atom (i)
                                                 j, # Second atom (j)
                                                 False, # Absolute distances
                                                 cutoff, # Distance
                                                 10**6, # (arbitrarily large)
                                                 force*HBD_HBA_f, # Larger constant
                                                 )
                # HBD/HBA-R Repulsion
                elif cond3:
                    # Add distance constraint (10.1186/s13321-014-0037-3)
                    ff.MMFFAddDistanceConstraint(i, # First atom (i)
                                                 j, # Second atom (j)
                                                 False, # Absolute distances
                                                 cutoff, # Distance
                                                 10**6, # (arbitrarily large)
                                                 force*HBC_R_f, # Larger constant
                                                 )     
                # R-R Repulsion
                else:
                    # Add distance constraint (10.1186/s13321-014-0037-3)
                    ff.MMFFAddDistanceConstraint(i, # First atom (i)
                                                 j, # Second atom (j)
                                                 False, # Absolute distances
                                                 cutoff, # Distance
                                                 10**6, # (arbitrarily large)
                                                 force*R_R_f, # Force constant
                                                 )
    # Output
    return ff

def isHB(molecule,index):
    """
    isHB() checks whether the atom of "molecule" with index "index" is an HBA
    or an HBD.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest.
    index : int
        Index of the atom

    Returns
    -------
    ans : string or None
        ans can take three values:
            'HBD' - The atom is a hydrogen bond donor
            'HBA' - The atom is a hydrogen bond acceptor
            None -  The atom is not an HBA or HBD

    """
    # Define answer as None
    ans=None
    # Get atom of interest
    atom=molecule.GetAtomWithIdx(index)
    # Get element of the atom
    element=atom.GetSymbol()
    # Define HBA atoms
    HBAs=['O','N','S','P']
    # Check if element is HBA
    if element in HBAs: ans='HBA'
    # Check if atom can be HBD (is proton?)
    if element=='H':
        # Get atoms bonded to atom of interest
        neighbors=atom.GetNeighbors()
        # Loop over neighbors
        for neighbor in neighbors:
            # If neighbor is HBA, then H is HBD
            if neighbor.GetSymbol() in HBAs: ans='HBD'
    # Output
    return ans

def internalHBs(molecule):
    """
    internalHBs() checks whether a molecule is capable of forming internal
    hydrogen bonds. This is done by counting the number HBD and HBA atoms in
    the molecule, and checking if there is at least 1 HBD and at least 2 HBAs
    (Because 1 HBA is by definition connected to the hydrogen that is the HBD).

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest.

    Returns
    -------
    ans : Boolean
        ans can take 2 values:
            'True' - The molecule is capable of forming internal hydrogen bonds
            'False' - The molecule is not capable of forming internal hydrogen

    """
    # Get number of atoms
    n_atoms = molecule.GetNumAtoms()
    # Initialize number of HBD and HBA counters
    num_HBD = 0
    num_HBA = 0
    # loop over all atoms and update counters
    for a in range(n_atoms):
        atm_type = isHB(molecule, a)
        if atm_type == 'HBD':
            num_HBD += 1
        elif atm_type == 'HBA':
            num_HBA += 1
    num_HBC = num_HBD + num_HBA
    # Check for possibility of internal hydrogen bonds
    if num_HBD > 1 and num_HBA > 2:     # 1 HBA will be the electronegative atom bonded to H, 
                                        # the other will be an available H-bonding site 
        ans = True
    else:
        ans = False
    # Output
    return ans

def getNeighborsTree(molecule,index,depth):
    """
    getNeighborsTree() gets the indexes of all atoms that are 1-2, 1-3, or 1-4
    neighbors of the atom with index "index".

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object
        Molecule object of interest.
    index : int
        Index of the atom
    depth : int
        Depth of the list. One of:
            2 - Return 1-2 neighbors
            3 - Reuturn 1-2 and 1-3 neighbors
            4 - Return 1-2, 1-3, and 1-4 neighbors

    Returns
    -------
    tree : list of ints
        List with the indexes of all the atoms that are neighbors of the atom
        with index "index", with the requested depth.

    """
    tree=[]
    # Get atom of interest
    atom=molecule.GetAtomWithIdx(index)
    # Get 12 Neighbors
    neighbors12=atom.GetNeighbors()
    # Iterate over 12 neighbors
    for neighbor12 in neighbors12:
        # Add index of neighbor12 to list
        tree.append(neighbor12.GetIdx())
        if depth>2:
            # Get 13 Neighbors
            neighbors13=neighbor12.GetNeighbors()
            # Iterate over 13 neighbors
            for neighbor13 in neighbors13:
                # Add index of neighbor13 to list
                tree.append(neighbor13.GetIdx())
                if depth>3:
                    # Get 14 Neighbors
                    neighbors14=neighbor13.GetNeighbors()
                    # Iterate over 13 neighbors
                    for neighbor14 in neighbors14:
                        # Add index of neighbor14 to list
                        tree.append(neighbor14.GetIdx())
    # Output
    return tree