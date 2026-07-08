# -*- coding: utf-8 -*-
"""
spGenerator is the main library of this Python package. Relying on wrapper
packages, namely RDKit_Wrapper.py and NWChem_Wrapper, its main function is to
employ the sigma profile generation algorithm developed in this work.

Sections
    . Imports
    
    . Main Functions
        . generateSP()
        . benchmarkPerformance()
        . benchmarkTessellation()

    . Auxiliary Functions
        . crossCheck()
        . getFragments()
        . getSigmaMatrix()
        . averagingAlgorithm()
        . getSigmaProfile()
        . combineFragmentSPs()
        . extractEnergyProfiles()
        . extractFinalEnergy()

Last edit: 2026-06-23
Author: Dinis Abranches, Fathya Salih
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import time
import secrets
import shutil
import random
import glob
from pathlib import Path

# Specific
import numpy
import pandas
import cirpy
import pubchempy
from rdkit import Chem
from rdkit.Chem import rdmolops

# Local
from lib import RDKit_Wrapper as rdk   # seeded initial conformer generation
from lib import NWChem_Wrapper as nwc

# =============================================================================
# Main Functions
# =============================================================================
    
def generateSP(identifier,jobFolder,np,configFile,logPath,
               identifierType='SMILES',
               charge=None,
               initialXYZ=None,
               randomSeed=42,
               cleanOutput=True,
               removeNWOutput=True,
               generateFinalXYZ=True,
               generateOutputSummary=True,
               doCOSMO=True,
               avgRadius=0.5,
               sigmaBins=[-0.250,0.250,0.001]):
    """
    generateSP() is the main function of the workflow developed to generate
    consistent sigma profiles. Given an identifier of a molecule, this function
    is responsible for:
        1. Converting the identifier into a SMILES string
        2. Obtaining an initial XYZ structure of the molecule using the custom
           MMFF force field implemented in RDKit_Wrapper
        3. Running NWChem using the NWChem_Wrapper and configuration file
        4. Reading COSMO results and compute the sigma profile of the molecule
    
    Alternatively, thus function can also be used to generate an optimized
    conformer for a molecule (using RDKit_Wrapper + quick HF NWChem run). This
    serves as the starting point for further COSMO calculations and is useful
    when building several sigma profile databases where the compounds are the
    same and only QM level of theory changes are made (e.g., different basis
    sets, different functionals, different tessellation, etc.)
    
    Files generated:
        

    Parameters
    ----------
    identifier : string
        Molecule identifier.
    jobFolder : string
        Path to the folder where all intermediate and final results are stored.
    np : int
        Number of threads to be used by NWChem (mpirun -np np ...).
        NOTE: cannot use np=1.
    configFile : string
        Path to the nwchem configuration file. See /path/to/lib/_config.
    logPath : string
            Path to log file.
    identifierType : string, optional
        Type of molecule identifier. One of:
            . 'SMILES'
            . 'CAS Number'
            . 'InChI'
            . 'InChIKey'
        The default is 'SMILES'.
    charge : int or None, optional
        Charge of the molecule/ion. If None, crossCheck() and RDKit_Wrapper are
        used to calculate it.
        The default is None.
    initialXYZ : string or None, optional
        If None, an initial xyz structure for the molecule will be generated
        using the custom MMFF force field as implemented in RDKit_Wrapper.
        If a path to an initial xyz is provided, this will be used and
        RDKit_Wrapper is bypassed.
        The default is None.
    cleanOutput : boolean, optional
        Whether to remove unnecessary output NWCHem files from the job folder,
        The default is True.
    removeNWOutput : boolean, optional
        Whether to remove the main log file from the job folder.
        The default is True.
    generateFinalXYZ : boolean, optional
        Whether to generate an xyz file with the final structure of the
        molecule after the geometry optimization in NWChem.
        The default is True.
    generateOutputSummary : boolean, optional
        Whether to copy and store the last step of the (big) log file of
        NWChem.
        The default is True.
    doCOSMO : boolean, optional
        Whether COSMO-related calculations were requested from NWChem. If TRUE,
        the function will expect COSMO-related information in the output files
        of NWCHem and will calculate a sigma profile.
        The default is True.
    avgRadius : float or None, optional
        Average radius (in Angstroms) to use in the averaging algorithm. If
        None, the averaging algorithm is not used.
        The default is 0.5.
    sigmaBins : list of floats, optional
        List containing information about the binning procedure for the sigma
        profile:
            sigmaBins[0] - Central coordinate of the first bin
            sigmaBins[1] - Central coordinate of the last bin
            sigmaBins[2] - Step between the centers of each bin
        The default is [-0.250,0.250,0.001].

    Returns
    -------
    warning : string or None
        String containing a warning raised by cross check or None if no
        warnings were raised.

    """
    # Initialize warning
    warning=None
    # If charge or initialXYZ are not provided, retrieve mol SMILES string 
    # (for calculating charge and generating geometry, respectively)
    if charge is None or initialXYZ in [None, 'RAND', 'RANDOM']:
        # If identifier is not a SMILES string, obtain a SMILES string
        if identifierType.upper() not in ['SMILES', 'MOL2']: 
            smilesString,warning=crossCheck(identifier,identifierType)
        else:
            smilesString=identifier
    # If SMILES string is for a compound, get all fragments
    smilesList = getFragments(smilesString)
    # Save jobFolder before string is split into fragments
    baseJobFolder=os.path.abspath(str(jobFolder))
    # Loop over fragments
    for s, fragSmiles in enumerate(smilesList):
        # Create job folder for current fragment
        jobFolder=os.path.join(baseJobFolder,f'fragment_{s}')
        os.makedirs(jobFolder, exist_ok=True)
        with open(logPath,'a') as logFile:
            logFile.write(f'\nProcessing fragment {s}/{len(smilesList)} with SMILES: {fragSmiles}')
        # Create path to initial conformer for the molecule
        xyzPath=os.path.join(jobFolder,'initialGeometry.xyz')
        if identifierType.upper()=='MOL2': # generate a pre-optimized version of the provided geometry
            molecule=rdk.moleculeFromMol2(identifier,xyzPath=xyzPath)
        elif initialXYZ is None: # generate an algorithm-selected conformer
            molecule=rdk.generateConformer(fragSmiles,xyzPath=xyzPath)
        elif initialXYZ == 'Random': # generate a random conformer
            molecule=rdk.getInitialConformer(fragSmiles,randomSeed=randomSeed,xyzPath=xyzPath)
        else: # Copy supplied xyz file to job folder as initialGeometry.xyz
            shutil.copy2(initialXYZ,xyzPath)
        # Get formal charge of molecule
        if charge is None: 
            charge=rdmolops.GetFormalCharge(molecule)
            with open(logPath,'a') as logFile:
                logFile.write('\tGiven charge was None, calculated charge: '+str(charge)+'\n')
        # Use job folder name as job name
        name=os.path.basename(os.path.normpath(jobFolder))
        # Generate NWChem input script
        inputPath=os.path.join(jobFolder,'input.nw')
        nwc.buildInputFile(inputPath,configFile,xyzPath,name,charge)
        # Run NWChem
        nwc.runNWChem(inputPath,jobFolder,np)
        # Check that nwchem job converged
        outputPath=os.path.join(jobFolder,'output.nw')
        converged=nwc.checkConvergence(outputPath)
        if converged != 1: removeNWOutput=False; generateFinalXYZ=True
        # Read cosmo.xyz
        if doCOSMO:
            cosmoPath=os.path.join(jobFolder,f'{name}.cosmo.xyz')
            segmentCoordinates,segmentCharges=nwc.readCOSMO(cosmoPath)
        # Read output file
        surfaceArea,segmentAreas,atomCoords,segAtoms=nwc.readOutput(outputPath,doCOSMO)
        # Generate final XYZ
        if generateFinalXYZ: 
            nwc.generateFinalXYZ(atomCoords,
                                os.path.join(jobFolder,'finalGeometry.xyz'))
        
        # Generate output summary
        if generateOutputSummary: 
            nwc.generateLastStep(outputPath,
                                os.path.join(jobFolder,'outputSummary.nw'))
        # Clean output
        if cleanOutput:
            for file in glob.glob(os.path.join(jobFolder,name+'*')):
                if 'cosmo' not in file: os.remove(file)
        # Remove NWChem log file
        if removeNWOutput:
            os.remove(outputPath)
        # Do COSMO
        if doCOSMO and converged >= 0:
            # Get sigmaMatrix
            sigmaMatrix,avgSigmaMatrix=getSigmaMatrix(segmentCoordinates,
                                                    segmentCharges,
                                                    segmentAreas,
                                                    surfaceArea,
                                                    segAtoms,
                                                    avgRadius=avgRadius,
                                                    logPath=logPath)  
            # Write non-averaged sigmaMatrix
            spPath=os.path.join(jobFolder,f'sigmaSurface.csv')
            numpy.savetxt(spPath,
                        sigmaMatrix,
                        delimiter=',')        
            # Get Sigma Profile
            sigma,sigmaProfile=getSigmaProfile(avgSigmaMatrix,sigmaBins)
            # Write Sigma Profile
            spPath=os.path.join(jobFolder,f'sigmaProfile.csv')
            numpy.savetxt(spPath,
                        numpy.column_stack((sigma,sigmaProfile)),
                        delimiter=',')
        # Raise NWChem errors, if any
        if converged == 0:
            raise Exception('NWChem job failed to converge in COSMO solvation medium, but converged in vacuum.'
                            +'\n\tThe full output.nw file along with final configuration will be returned...')
        elif converged == -1:
            raise Exception('NWChem job failed to converge in vacuum. Optimization in COSMO solvation medium was not attempted.'
                            +'\n\tThe full output.nw file along with the final configuration will be returned...')

    # Combine fragment sigma profiles, if needed
    if len(smilesList) > 1:
        spPath=os.path.join(baseJobFolder,f'sigmaProfile.csv')
        sigma,combSP=combineFragmentSPs(baseJobFolder,len(smilesList),sigmaBins)
        numpy.savetxt(spPath,numpy.column_stack((sigma,combSP)),
                                delimiter=',')
    else:
        # no combination required, copy files as they are
        sourcePaths=[]
        targetPaths=[]
        if not removeNWOutput: 
            sourcePaths =+ os.path.join(jobFolder,'output.nw')
            targetPaths =+ os.path.join(baseJobFolder,'output.nw')
        if generateFinalXYZ:
            sourcePaths =+ os.path.join(jobFolder,'finalGeometry.xyz')
            targetPaths =+ os.path.join(baseJobFolder,'finalGeometry.xyz')
        if doCOSMO:
            sourcePaths =+ [
                os.path.join(jobFolder,f'{name}.cosmo.xyz'),
                os.path.join(jobFolder,'sigmaSurface.csv'),
                os.path.join(jobFolder,'sigmaProfile.csv'),
            ]
            targetPaths =+ [
                os.path.join(baseJobFolder,f'{name}.cosmo.xyz'),
                os.path.join(baseJobFolder,'sigmaSurface.csv'),
                os.path.join(baseJobFolder,'sigmaProfile.csv'),
            ]
        if generateOutputSummary: 
            sourcePaths =+ os.path.join(jobFolder,'outputSummary.nw')
            targetPaths =+ os.path.join(baseJobFolder,'outputSummary.nw')           
        for sourcePath, targetPath in zip(sourcePaths,targetPaths):
            shutil.copy2(sourcePath,targetPath)

    # Output
    return warning

# =============================================================================
# Auxiliary Functions
# =============================================================================

def crossCheck(identifier,identifierType):
    """
    crossCheck() obtains the SMILES string of a compound described by
    "identifier" using two different databases (CIRpy and PubChemPy). If the
    SMILES strings match, they are returned. If not, a warning is raised and
    the SMILES string from PubChemPy is used. If neither database return a hit,
    an exception is raised.

    Parameters
    ----------
    identifier : string
        Molecule identifier.
    identifierType : string
        Type of molecule identifier. One of:
            . 'CAS Number'
            . 'InChI'
            . 'InChIKey'

    Raises
    ------
    Exception
        When the identifier cannot be found in CIRpy and PubChemPy.

    Returns
    -------
    smilesString : string
        SMILES string of the molecule.
    warning : string or None
        If cross check failed, a warning is returned.

    """
    # Initialize warning
    warning=None
    # Obtain SMILES string using CIRpy (identifier type infered automatically)
    for __ in range(10): # Protect against random connection errors
        try:
            # Returns None if identifier is not found
            smilesString_1=cirpy.resolve(identifier,'smiles')
            break
        except:
            smilesString_1=None
            time.sleep(10)
    # Get PubChem identifier type
    if identifierType=='CAS-Number': pubType='name'
    if identifierType=='InChI': pubType='inchi'
    if identifierType=='InChIKey': pubType='inchikey'
    # Obtain SMILES string using PubChemPy 
    for __ in range(10): # Protect against random connection errors
        try:
            # Returns empty list if identifier is not found
            smilesString_2=pubchempy.get_compounds(identifier,pubType)[0].isomeric_smiles
            break
        except:
            smilesString_2=None
            time.sleep(10)
    # Cross check SMILES strings
    if smilesString_1 is None and not smilesString_2:
        # If identifier could not be found in neither database, raise exception
        raise ValueError('Could not find identifier provided...')
    elif not smilesString_2:
        # If identifier could not be found in PubChemPy, add warning
        warning='Identifier not found by PubChemPy...'
        # Set smiles as those returned by CIRpy
        smilesString=smilesString_1
    elif smilesString_1 is None:
        # If identifier could not be found in CIRpy, add warning
        warning='Identifier not found by CIRpy...'
        # Set smiles as those returned by PubChemPy
        smilesString=smilesString_2
    else:
        # Canonicalize smiles with RDKit before comparing
        mol1=Chem.MolFromSmiles(smilesString_1)
        mol2=Chem.MolFromSmiles(smilesString_2)
        smilesString_1=Chem.rdmolfiles.MolToSmiles(mol1)
        smilesString_2=Chem.rdmolfiles.MolToSmiles(mol2)
        # Cross check
        if smilesString_1!=smilesString_2:
            # If smiles do not match, add warning
            warning='Failed to find the SMILES string in PubChem and ...'
        # Set smiles as those returned by PubChemPy
        smilesString=smilesString_2
    # Output
    return smilesString,warning

def getFragments(smilesString):
    """
    getFragments() checks if the SMILES string of a compound represents a compound
    with fragments like a salt or a mixture (e.g. [Cl-].[Cl-].[F-].[Fe+3]). A list
    of SMILES strings is returned, even if the molecule contains only 1 fragment.

    Parameters
    ----------
    smilesString : string
        SMILES string of the molecule.

    Raises
    ------
    ValueError
        ValueError is raised if the provided SMILES string is invalid.

    Returns
    -------
    smilesList : List, string
        List of SMILES strings of each fragment in the molecule/compound.
    """
    # Parse SMILES safely
    mol = Chem.MolFromSmiles(smilesString)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Get fragments as tuple of molecule objects
    fragments = rdmolops.GetMolFrags(mol, asMols=True)

    # Convert fragments/Mol objects to SMILES strings
    smilesList = [Chem.MolToSmiles(frag) for frag in fragments]

    return smilesList

    
def getSigmaMatrix(segmentCoordinates,segmentCharges,segmentAreas,surfaceArea,segAtoms,
                   avgRadius=None,logPath=None):
    """
    getSigmaMatrix() computes the sigma matrix of the molecule.

    Parameters
    ----------
    segmentCoordinates : list of list of floats
        List where each entry corresponds to a list with the coordinates
        (x,y,z) of a segment.
    segmentCharges : list of floats
        List where each entry corresponds to the charge of a segment.
    segmentAreas : list of floats
        List with the surface area of each segment (a.u.^2).
    surfaceArea : float
        Total surface area of the molecule (Ang^2).
    segAtoms : list of floats
        Atoms to which segments belong
    avgRadius : float or None
        Average radius to use in the averaging algorithm. If None, the
        averaging algorithm is not used.
    logPath : string or None
        Path to the log file. If None, no log file is written.

    Raises
    ------
    ValueError
        ValueError is raised if the sum of the areas of the segments does not
        match the total surface area of the molecule retrieved from NWChem.

    Returns
    -------
    sigmaMatrix : numpy.ndarray of floats
        Matrix containing sigma surface information:
            . Column 0 - x coordinate of point charge (Angs)
            . Column 1 - y coordinate of point charge (Angs)
            . Column 2 - z coordinate of point charge (Angs)
            . Column 3 - charge of point charge (e)
            . Column 4 - area of surface segment (Angs^2)
            . Column 5 - charge density of segment (e/Angs^2)
            . Column 6 - atom index to which segment belongs

    avgSigmaMatrix : numpy.ndarray of floats
        Matrix containing sigma surface information, with column 5 recalculated
        using the averaging algorithm:
            . Column 0 - x coordinate of point charge (Angs)
            . Column 1 - y coordinate of point charge (Angs)
            . Column 2 - z coordinate of point charge (Angs)
            . Column 3 - charge of point charge (e)
            . Column 4 - area of surface segment (Angs^2)
            . Column 5 - charge density of segment (e/Angs^2)
            . Column 6 - atom index to which segment belongs
    """
    # Get total number of segments
    nSeg=len(segmentCharges)
    # Generate empty sigmaMatrix
    sigmaMatrix=numpy.zeros([nSeg,7])
    # Structure of sigmaMatrix (each line represents a surface segment)
    #   . Column 0 - x coordinate of point charge (Angs)
    #   . Column 1 - y coordinate of point charge (Angs)
    #   . Column 2 - z coordinate of point charge (Angs)
    #   . Column 3 - charge of point charge (e)
    #   . Column 4 - area of surface segment (Angs^2)
    #   . Column 5 - charge density of segment (e/Angs^2)
    #   . Column 6 - atom index to which segment belongs
    # Fill out sigmaMatrix
    for n in range(nSeg):
        # x coordinate of point charge n (Angs)
        sigmaMatrix[n,0]=segmentCoordinates[n][0] 
        # y coordinate of point charge n (Angs)
        sigmaMatrix[n,1]=segmentCoordinates[n][1]
        # z coordinate of point charge n (Angs)
        sigmaMatrix[n,2]=segmentCoordinates[n][2]
        # charge of point charge n (e)
        sigmaMatrix[n,3]=segmentCharges[n]
        # area of surface segment n (Angs^2)
        sigmaMatrix[n,4]=segmentAreas[n]*(0.529177249**2)
        # charge density of segment k (e/Angs^2)  
        sigmaMatrix[n,5]=sigmaMatrix[n,3]/sigmaMatrix[n,4] 
        # atom index to which segment belongs
        sigmaMatrix[n,6]=segAtoms[n]
        # if NaN is encountered, raise error
        if numpy.isnan(sigmaMatrix[n,:]).any():
            with open(logPath,'a') as logFile:
                logFile.write(f'\nNaN value encountered in sigma matrix, segment number {n+1}...')
            raise ValueError('NaN value encountered in sigma matrix...')
    # Check that the total area calculated by NWChem matches
    # the sum of the areas of the segments
    if abs(sum(sigmaMatrix[:,4])-surfaceArea)>0.1:
        raise ValueError('Surface area inconsistency...')
    # Perform averaging algorithm, if requested
    if avgRadius is not None:
        avgSigmaMatrix=averagingAlgorithm(sigmaMatrix,avgRadius)
    else:
        avgSigmaMatrix=sigmaMatrix
    # Output
    return sigmaMatrix,avgSigmaMatrix

def averagingAlgorithm(sigmaMatrix,avgRadius):
    """
    Perform an averaging algorithm on the sigma surface.

    Parameters
    ----------
    sigmaMatrix : numpy.ndarray of floats
        Matrix containing sigma surface information:
            . Column 0 - x coordinate of point charge (Angs)
            . Column 1 - y coordinate of point charge (Angs)
            . Column 2 - z coordinate of point charge (Angs)
            . Column 3 - charge of point charge (e)
            . Column 4 - area of surface segment (Angs^2)
            . Column 5 - charge density of segment (e/Angs^2)
            . Column 6 - atom index to which segment belongs
    avgRadius : float
        Average radius to use in the averaging algorithm.

    Returns
    -------
    sigmaMatrix : numpy.ndarray of floats
        Matrix containing sigma surface information, with column 5 recalculated
        using the averaging algorithm:
            . Column 0 - x coordinate of point charge (Angs)
            . Column 1 - y coordinate of point charge (Angs)
            . Column 2 - z coordinate of point charge (Angs)
            . Column 3 - charge of point charge (e)
            . Column 4 - area of surface segment (Angs^2)
            . Column 5 - charge density of segment (e/Angs^2)
            . Column 6 - atom index to which segment belongs

    """
    # Get squared avaraging radius
    sqRav=avgRadius**2
    # Get vector with squared radii
    sqR=sigmaMatrix[:,4]/numpy.pi
    # Initialize container for averaged sigmas
    avgSigma=numpy.zeros([sigmaMatrix.shape[0],])
    # First loop over segments
    for i in range(sigmaMatrix.shape[0]):
        # Get vector with squared distance between i and all other segments
        d=((sigmaMatrix[i,0]-sigmaMatrix[:,0])**2
           +(sigmaMatrix[i,1]-sigmaMatrix[:,1])**2
           +(sigmaMatrix[i,2]-sigmaMatrix[:,2])**2)
        # Calculate denominator vector
        denVector=((sqR*sqRav)/(sqR+sqRav))*numpy.exp(-d/(sqR+sqRav))
        # Calculate numerator vector
        numVector=sigmaMatrix[:,5]*denVector
        # Update avgSigma
        avgSigma[i]=numVector.sum()/denVector.sum()
    sigmaMatrix[:,5]=avgSigma
    # Output
    return sigmaMatrix
    
def getSigmaProfile(sigmaMatrix,sigmaBins):
    """
    getSigmaProfile() calculates the sigma profile of the molecule described by
    sigmaMatrix.

    Parameters
    ----------
    sigmaMatrix : numpy.ndarray of floats
        Matrix containing sigma surface information:
            . Column 0 - x coordinate of point charge (Angs)
            . Column 1 - y coordinate of point charge (Angs)
            . Column 2 - z coordinate of point charge (Angs)
            . Column 3 - charge of point charge (e)
            . Column 4 - area of surface segment (Angs^2)
            . Column 5 - charge density of segment (e/Angs^2)
            . Column 6 - atom index to which segment belongs
    sigmaBins : list of floats
        List containing information about the binning procedure for the
        sigma profile:
            sigmaBins[0] - Central coordinate of the first bin
            sigmaBins[1] - Central coordinate of the last bin
            sigmaBins[2] - Step between the centers of each bin

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    sigma : list of floats
        List containing the sigma bins (e/Ang^2)
    sp : list of floats
        List containing the sigma profile values for each sigma bin (Ang^2).

    """
    # Check that the sigmaSurface values are within the sigma range
    condMin=min(sigmaMatrix[:,5])<sigmaBins[0]
    condMax=max(sigmaMatrix[:,5])>sigmaBins[1]
    if condMin or condMax:
        raise ValueError('Sigma values outside of range...')
    # Generate bins (location of bin center, sigma vector)
    sigma=numpy.arange(sigmaBins[0],sigmaBins[1]+sigmaBins[2],sigmaBins[2])
    # Initialize sigma profile vector
    sp=numpy.zeros(len(sigma))
    # Loop over sigma surface
    for n in range(sigmaMatrix.shape[0]):
        
        i_left=int(numpy.floor((sigmaMatrix[n,5]-sigmaBins[0])/sigmaBins[2]))
        w=(sigma[i_left+1]-sigmaMatrix[n,5])/sigmaBins[2]
        sp[i_left]+=w*sigmaMatrix[n,4]
        sp[i_left+1]+=(1-w)*sigmaMatrix[n,4]

    # Output
    return sigma,sp

def combineFragmentSPs(baseJobFolder, nFragments, sigmaBins):
    """
    Combine SPs from different fragments.

    Parameters
    ----------
    baseJobFolder : os.path object
        Path to base folder outside the fragment job folders
    nFragments : int
        Number of fragments in provided identifier SMILES string
    sigmaBins : list of floats
        List containing information about the binning procedure for the
        sigma profile:
            sigmaBins[0] - Central coordinate of the first bin
            sigmaBins[1] - Central coordinate of the last bin
            sigmaBins[2] - Step between the centers of each bin

    Returns
    -------
    sigma : list of floats
        List containing the sigma bins (e/Ang^2)
    sp_comb : list of floats
        List containing the combined sigma profile values for each sigma
        bin (Ang^2).
    """
    sigma=numpy.arange(sigmaBins[0],sigmaBins[1]+sigmaBins[2],sigmaBins[2])
    nbins=len(sigma)
    # initialize array for all fragemnt SPs
    sp_arr = numpy.zeros((nbins,nFragments))
    # loop over fragments
    for s in range(nFragments):
        # get job folder for current fragment
        jobFolder = os.path.join(baseJobFolder,f'fragment_{s}')
        # read fragment SP
        sp_file = os.path.join(jobFolder,'sigmaProfile.csv')
        sp_frag = pandas.read_csv(sp_file, header=None).to_numpy()
        sp_arr[:,s] = sp_frag[:,1]
    # average/combine different fragment SPs
    sp_comb = numpy.mean(sp_arr, axis=1)  
    return sigma,sp_comb

def extractEnergyProfiles(output_file, doCOSMO=True):
    """
    extract_energies() extracts the energy profiles from the complete
    output.nw file during different stages of optimization for plotting.

    Parameters
    ----------
    output_file : str
        path to complete nwchem output file
    doCOSMO : boolean, optional
        Whether COSMO-related calculations were requested from NWChem. If TRUE,
        the function will expect COSMO-related information in the output files
        of NWCHem and will calculate a sigma profile.
        The default is True.
    Returns
    -------
    hf_steps : list of floats
        List containing initial vacuum HF optimization step numbers
    hf_energies : list of floats
        List containing initial vacuum HF optimization energies
    dft_steps : list of floats
        List containing optimization step numbers fom the vacuum 
            geometry optimization performed at the desired level of 
            theory.
    dft_energies : list of floats
        List containing energies fom the vacuum geometry optimization 
            performed at the desired level of theory. Returns list of 
            nans if no DFT optimization was done in vacuum.
    cosmo_steps : list of floats
        List containing optimization step numbers fom the COSMO 
            geometry optimization performed at the desired level of 
            theory.
    cosmo_energies : list of floats
        List containing energies fom the COSMO geometry optimization 
            performed at the desired level of theory. Returns list of
            nans if no COSMO optimization was done.
    """
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
        first_hf_en_lines = numpy.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
        rest_hf_en_lines = numpy.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
        all_hf_en_lines = numpy.concatenate((first_hf_en_lines,rest_hf_en_lines))
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
        hf_energies = numpy.array(hf_energies)
        hf_steps = numpy.array(hf_steps)

        ## Go to DFT section
        if not no_dft:
            # get DFT energy lines
            first_dft_en_lines = numpy.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
            rest_dft_en_lines = numpy.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
            all_dft_en_lines = numpy.concatenate((first_dft_en_lines,rest_dft_en_lines))
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
            dft_energies = numpy.array(dft_energies)
            dft_steps = numpy.array(dft_steps)
        else:
            dft_energies = numpy.zeros_like(hf_energies) * numpy.nan
            dft_steps = numpy.zeros_like(hf_steps)

        ## Go to COSMO section
        if doCOSMO:
            # get COSMO energy lines
            first_cosmo_en_lines = numpy.array(nwc.findAllOccurrences(file,first_energy_header)) + 2
            rest_cosmo_en_lines = numpy.array(nwc.findAllOccurrences(file,rest_energy_headers)) + 2
            all_cosmo_en_lines = numpy.concatenate((first_cosmo_en_lines,rest_cosmo_en_lines))
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
            cosmo_energies = numpy.array(cosmo_energies)
            cosmo_steps = numpy.array(cosmo_steps)
        else:
            cosmo_energies = numpy.zeros_like(hf_energies) * numpy.nan
            cosmo_steps = numpy.zeros_like(hf_steps)

    return (hf_steps, hf_energies), (dft_steps, dft_energies), (cosmo_steps, cosmo_energies)

def extractFinalEnergy(output_summary_file):
    """
    extract_energies() extracts the final DFT/COSMO energy from the
    output summary file.

    Parameters
    ----------
    output_summary_file : str
        path to nwchem output summary file generated by OpenSPGen

    Returns
    -------
    final_energy : float
        last energy printed in the summary file
    """
    with open(output_summary_file, 'r') as file:
        # read lines
        lines = file.readlines()

        # get lines containing energy header
        energy_header = 'Step Energy Delta E Gmax Grms Xrms Xmax Walltime'.split()
        header_lines_idx = numpy.array(nwc.findAllOccurrences(file, energy_header))
        energy_lines_idx = header_lines_idx + 2

        # loop over lines (should be 1) and save last energy
        for l in energy_lines_idx:
            line = lines[l].split()
            if '@' != line[0]: final_energy = float(line[1].replace('D', 'E'))
            else: final_energy = float(line[2].replace('D', 'E'))

        return final_energy
