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
        . getSigmaMatrix()
        . averagingAlgorithm()
        . getSigmaProfile()

Last edit: 2022-06-14
Author: Dinis Abranches
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

# Specific
import numpy
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
    
def generateSP(identifier,jobFolder,np,configFile,
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
        The default is [-0.035,0.035,0.001].

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
    if charge is None or initialXYZ in [None, 'Random']:
        # If identifier is not a SMILES string, obtain a SMILES string
        if identifierType.upper() not in ['SMILES', 'MOL2']: 
            smilesString,warning=crossCheck(identifier,identifierType)
        else:
            smilesString=identifier
    # Create path to initial conformer for the molecule
    xyzPath=os.path.join(jobFolder,'initialGeometry.xyz')
    if identifierType.upper()=='MOL2': # generate a pre-optimized version of the provided geometry
        molecule=rdk.moleculeFromMol2(identifier,xyzPath=xyzPath)
    elif initialXYZ is None: # generate an algorithm-selected conformer
        molecule=rdk.generateConformer(smilesString,xyzPath=xyzPath)
    elif initialXYZ == 'Random': # generate a random conformer
        molecule=rdk.getInitialConformer(smilesString,randomSeed=randomSeed,xyzPath=xyzPath)
    else: # Copy supplied xyz file to job folder as initialGeometry.xyz
        shutil.copy2(initialXYZ,xyzPath)
    # Get formal charge of molecule
    if charge is None: 
        charge=rdmolops.GetFormalCharge(molecule)
        with open(logPath,'a') as logFile:
            logFile.write('\nGiven charge was None, calculated charge: '+str(charge)+'\n')
    # Use job folder name as job name
    name=os.path.basename(os.path.normpath(jobFolder))
    # Generate NWChem input script
    inputPath=os.path.join(jobFolder,'input.nw')
    nwc.buildInputFile(inputPath,configFile,xyzPath,name,charge)
    # Run NWChem
    nwc.runNWChem(inputPath,jobFolder,np)
    # Check that nwchem job converged
    outputPath=os.path.join(jobFolder,'output.nw')
    logPath=os.path.join(jobFolder,'..','job.log')
    converged=nwc.checkConvergence(outputPath)
    if converged != 1: removeNWOutput=False; generateFinalXYZ=True
    # Read cosmo.xyz
    if doCOSMO:
        cosmoPath=os.path.join(jobFolder,name+'.cosmo.xyz')
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
        os.remove(os.path.join(jobFolder,'output.nw'))
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
        spPath=os.path.join(jobFolder,'sigmaSurface.csv')
        numpy.savetxt(spPath,
                      sigmaMatrix,
                      delimiter=',')        
        # Get Sigma Profile
        sigma,sigmaProfile=getSigmaProfile(avgSigmaMatrix,sigmaBins)
        # Write Sigma Profile
        spPath=os.path.join(jobFolder,'sigmaProfile.csv')
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
    # Output
    return warning
    
def benchmarkPerformance(logPath,nRepetitions,npList,
                         # args passed to generateSP():
                         identifier,configFile,
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
    benchmarkPerformance() benchmarks the performance of generateSP()) as a
    function of the number of threads used. For each np in npList, the function
    calls generateSP() and registers its execution time. The benchmarks are
    performed inside the temporary folder of the package and all scratch and
    final files are removed. Order of npList and repetitions is shuffled to
    prevent biased results due to temporary CPU underperformance.
    
    Parameters
    ----------
    logPath : string
        Path to the output file.
    nRepetitions : int
        Number of times to repeat a benchmark for a given np.
        Useful for reproducibility.
    npList : list of ints
        List where each entry is a number of threads to benchmark.
        NOTE: cannot use np=1.
    **See generateSP() for remaining input arguments.

    Returns
    -------
    None.

    """
    # Create log file
    with open(logPath,'w') as logFile:
        logFile.write('Log file from benchmarkSingleRun().\n\n')
        logFile.write('Identifier: '+identifier+'\n\n')
        logFile.write('Results (np,t):'+'\n')
    # Generate new npList including number of repetitions
    npList=nRepetitions*npList
    # Shuffle list
    random.shuffle(npList)
    # Loop over number of threads to time
    for np in npList:
        # Generate random name for temporary folder
        randomName=secrets.token_hex(15)
        # Create temporary folder inside master temporary folder
        tempPath=os.path.join(os.path.dirname(__file__),'_temp',randomName)
        os.makedirs(tempPath)
        # Register time
        t1=time.time()
        # Call generateSP()
        generateSP(identifier,tempPath,np,configFile,
                   identifierType=identifierType,
                   charge=charge,
                   initialXYZ=initialXYZ,
                   randomSeed=randomSeed,
                   cleanOutput=cleanOutput,
                   removeNWOutput=removeNWOutput,
                   generateFinalXYZ=generateFinalXYZ,
                   generateOutputSummary=generateOutputSummary,
                   doCOSMO=doCOSMO,
                   avgRadius=avgRadius,
                   sigmaBins=sigmaBins)
        # Get elapsed time
        t=str(round(time.time()-t1,2))
        # Update log file
        with open(logPath,'a') as logFile: logFile.write('\n'+str(np)+','+t)
        # Delete temporary folder
        shutil.rmtree(tempPath)
    # Output
    return None

def benchmarkTessellation(jobFolder,tessellation,
                          # args passed to generateSP():
                          identifier,np,configFile,
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
    benchmarkTessellation() benchmarks the impact of tessellation on 
    generateSP(). For each tessellation level in "tesselation", the function
    calls generateSP(). The benchmarks are performed inside the temporary
    folder folder of the package and main results are stored in "jobFolder".
    A log file is provided. Note that the function makes a copy of configFile
    and changes its tessellation-related keywords throughout the run.
    
    Parameters
    ----------
    jobFolder : string
        Path to the folder where intermediate and final results are stored.
    tessellation : list of tuples
        List where each entry is a tessellation level to be tried. Each entry
        is a tuple; the first entry of the tuple is the NWChem ificos keyword
        while the second entry is the NWCHem minbem keyword.
    npList : list of ints
        List where each entry is a number of threads to benchmark.
        NOTE: cannot use np=1.
    **See generateSP() for remaining input arguments.

    Returns
    -------
    None.

    """
    logPath=os.path.join(jobFolder,'log.out')
    with open(logPath,'w') as logFile:
        logFile.write('Log file from benchmarkTessellation().\n\n')
        logFile.write('Identifier: '+identifier+'\n\n')
        logFile.write('Any errors will be printed below.\n')
    # Loop over tessellation
    for tess in tessellation:
        # Generate random name for new config file
        randomName=secrets.token_hex(15)
        newConfigPath=os.path.join(os.path.dirname(__file__),
                                   '_temp',
                                   randomName+'.config')
        # Open base config file
        with open(configFile,'r') as originalFile:
            # Open new config file
            with open(newConfigPath,'w') as newFile:
                # Loop over original file
                for line in originalFile:
                    if 'minbem' in line.split(): # Change minbem
                        newFile.write('  minbem '+str(tess[1])+'\n')
                    elif 'ificos' in line.split(): # Change ificos
                        newFile.write('  ificos '+str(tess[0])+'\n')
                    else: # Copy line
                        newFile.write(line)
        # Generate random name for temporary folder
        randomName=secrets.token_hex(15)
        # Create temporary folder inside master temporary folder
        tempFolder=os.path.join(os.path.dirname(__file__),'_temp',randomName)
        os.makedirs(tempFolder)
        # Call generateSP() with error handling
        try:
            generateSP(identifier,tempFolder,np,newConfigPath,
                       identifierType=identifierType,
                       charge=charge,
                       initialXYZ=initialXYZ,
                       randomSeed=randomSeed,
                       cleanOutput=cleanOutput,
                       removeNWOutput=removeNWOutput,
                       generateFinalXYZ=generateFinalXYZ,
                       generateOutputSummary=generateOutputSummary,
                       doCOSMO=doCOSMO,
                       avgRadius=avgRadius,
                       sigmaBins=sigmaBins)
        except Exception as error:
            with open(logPath,'a') as logFile:
                logFile.write('\nError for tessellation tuple: '
                              +str(tess)
                              +'\n')
                logFile.write(str(error)+'\n')
        # Copy sigma profile file to jobFolder
        shutil.copy2(os.path.join(tempFolder,'sigmaProfile.csv'),
                     os.path.join(jobFolder,
                                  str(tess[0])+'_'+str(tess[1])+'.csv'))
        # Delete temporary folder and config file
        shutil.rmtree(tempFolder)
        os.remove(newConfigPath)
    # Output
    return None
    
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
            time.sleep(10)
    # Get PubChem identifier type
    if identifierType=='CAS-Number': pubType='name'
    if identifierType=='InChI': pubType='inchi'
    if identifierType=='InChIKey': pubType='inchikey'
    # Obtain SMILES string using PubChemPy 
    for __ in range(10): # Protect against random connection errors
        try:
            # Returns empty list if identifier is not found
            smilesString_2=pubchempy.get_compounds(identifier,pubType)
            break
        except:
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
        smilesString=smilesString_2[0].isomeric_smiles
    else:
        # Canonicalize smiles with RDKit before comparing
        mol1=Chem.MolFromSmiles(smilesString_1)
        mol2=Chem.MolFromSmiles(smilesString_2[0].isomeric_smiles)
        smilesString_1=Chem.rdmolfiles.MolToSmiles(mol1)
        smilesString_2=Chem.rdmolfiles.MolToSmiles(mol2)
        # Cross check
        if smilesString_1!=smilesString_2:
            # If smiles do not match, add warning
            warning='Cross check failed...'
        # Set smiles as those returned by PubChemPy
        smilesString=smilesString_2
    # Output
    return smilesString,warning
    
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
