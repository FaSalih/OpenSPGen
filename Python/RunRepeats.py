
# =============================================================================
# Imports
# =============================================================================

# General
import os
import time
import sys
import argparse
import pandas as pd
import numpy as np

# Local
from lib import spGenerator as sp

# =============================================================================
# Configuration (user-provided options)
# =============================================================================

# Read named variables from command line
parser = argparse.ArgumentParser()

parser.add_argument('--identifier', required=True, help='The identifier. Can be a SMILES string, CAS number, InChI, InChIKey, or xyz file path.')
parser.add_argument('--identifier_type', required=True, help='Molecule identifier type. Can be: SMILES, CAS Number, InChI, InChIKey, or xyz', default='SMILES')
parser.add_argument('--initial_xyz', required=False, help='The initial XYZ file path. If not provided, the initial geometry will be randomly generated from the identifier.', default='Random')
parser.add_argument('--molecule_id', required=True, help='The molecule number or name for the job folder name.')
parser.add_argument('--cas_number', required=False, help='The CAS number of the molecule. Used to determine the NWChem config file.', default=None)
parser.add_argument('--random_seeds', required=False, help='The list of random seeds to use for each job', default=[42], type=int)
parser.add_argument('--n_jobs', required=False, help='The number of repeat jobs to run', default=1, type=int)
parser.add_argument('--n_slots', required=True, help='The number of cores to use for NWChem')
parser.add_argument('--charge', required=False, help='The charge of the molecule', default=0, type=int)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
identifier = args.identifier
identifierType = args.identifier_type
initialXYZ = (identifier if identifierType == 'xyz' else args.initial_xyz)
molecule_id = args.molecule_id
cas_number = args.cas_number
np_NWChem = args.n_slots
randomSeeds = args.random_seeds
n_jobs = args.n_jobs
charge = args.charge

# Ensure input consistency
if identifierType not in ['SMILES', 'CAS Number', 'InChI', 'InChIKey', 'xyz']:
    raise ValueError('Invalid identifier type. Must be one of: SMILES, CAS Number, InChI, InChIKey, or xyz')
if initialXYZ not in [None, 'Random'] and not os.path.exists(initialXYZ):
    raise ValueError('Initial XYZ file does not exist.')
if initialXYZ is not None and identifierType != 'xyz':
    raise ValueError('Initial XYZ can only be provided for xyz identifier .')

# Specify job name
if initialXYZ in ['Random', 'Rand']:
    job_name=f'SP-RandInitXYZ-Mol_{molecule_id}'
elif initialXYZ is None and identifierType != 'xyz':
    job_name=f'SP-NoInitXYZ-Mol_{molecule_id}'
elif initialXYZ is not None and identifierType == 'xyz':
    job_name=f'SP-GivenInitXYZ-Mol_{molecule_id}'

# NWChem Config file base name
nwchemConfig='COSMO_b3lyp_TZVP'
# Do COSMO? (= calculate sigma profile, not just sigma surface)
doCOSMO=True
# Other spGenerator.py options:
cleanOutput=True
generateFinalXYZ=True
generateOutputSummary=True
avgRadius=0.5
sigmaBins=[-0.035,0.035,0.001]

# List of CAS numbers that require noautoz:
noautoz=['101-68-8','110-65-6','501-65-5','503-17-3','506-77-4','627-21-4',
         '646-05-9','689-97-4','764-35-2','928-49-4','75-15-0','463-58-1']
# List of CAS numbers containing Iodine
iodine=['74-88-4','75-03-6','75-11-6','75-30-9','107-08-4','542-69-8',
        '591-50-4','638-45-9','7553-56-2','10034-85-2']

# =============================================================================
# Auxiliary Functions
# =============================================================================

def call_generateSP(entry,configFile):
    """
    call_generateSP() is a wrapper around sp.generateSP(). It exists to
    faciliate calling sp.generateSP() and to return information about each job.
    To avoid unecessarily heavy code, this function accesses variables outside
    of its scope, which are not global and are not changed inside it.

    Parameters
    ----------
    entry : list of strings (len=2)
        Molecule entry. The first entry is used to name the job folder and
        file results, while the second entry is the actual entry used
        in the SP generation process.

    Returns
    -------
    entry : string
        Molecule entry. Same as input.
    t : int
        Elapsed time for the execution of the job (seconds).
    errorOcurred : boolean
        Whether an error occured.

    """
    # Register time
    t1=time.time()
    # Define job folder inside Main Folder
    jobFolder=os.path.join(mainFolder,entry[0])
    # Make folder
    if not(os.path.exists(jobFolder)):
        os.mkdir(jobFolder)
    # Call generateSP() with error handling
    errorOcurred=0
    # Get initial xyz
    initialXYZ_=initialXYZ
    # if initialXYZ_ is not None:
    if initialXYZ_ not in [None, 'Random']:
        initialXYZ_=os.path.join('..',initialXYZ_)
    try:
        warning=sp.generateSP(entry[1],jobFolder,np_NWChem,configFile,
                              identifierType=identifierType,
                              charge=charge,
                              initialXYZ=initialXYZ_,
                              randomSeed=randomSeeds[n],
                              cleanOutput=cleanOutput,
                              generateFinalXYZ=generateFinalXYZ,
                              generateOutputSummary=generateOutputSummary,
                              doCOSMO=doCOSMO,
                              avgRadius=avgRadius,
                              sigmaBins=sigmaBins)
        if warning is not None:
            with open(logPath,'a') as logFile:
                logFile.write('\nWarning for molecule: '+entry[0])
                logFile.write('\nThe following warnings were detected:\n')
                logFile.write(warning)
    except Exception as error:
        with open(logPath,'a') as logFile:
            logFile.write('\nJob failed for molecule: '+entry[0])
            logFile.write('\nThe following errors were detected:\n')
            logFile.write(str(error))
        errorOcurred=True
    # Get elapsed time
    t=round(time.time()-t1,2)
    # Output
    return entry,t,errorOcurred

def printLogHeader():
    """
    printLogHeader() prints the details of the parallel job to the log file.
    To avoid unecessarily heavy code, this function accesses variables outside
    of its scope, which are not global and are not changed inside it.

    Returns
    -------
    None.

    """
    # Create log file
    with open(logPath,'w') as logFile:
        logFile.write('Initializing serial task...\n')
        logFile.write('\tMain folder: '+mainFolder+'\n')
        # logFile.write('\tMolecule list: '+identifierListPath+'\n')
        logFile.write('\tNumber of threads per job: '+str(np_NWChem)+'\n')
        logFile.write('\tNWChem configuration file: '+nwchemConfig+'\n')
        logFile.write('\tDo COSMO: '+str(doCOSMO)+'\n')
        if doCOSMO:
            logFile.write('\tAveraging radius: '+str(avgRadius)+'\n')
            logFile.write('\tSigma bins: '+str(sigmaBins)+'\n')
        logFile.write('Initialization complete.\n')
    # Output
    return None

# =============================================================================
# Main Script
# =============================================================================
# Get the basename of the job
basename=job_name
# Path to the main folder.
mainFolder=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '..',
                        basename)
# Make main folder
if not os.path.isdir(mainFolder): os.makedirs(mainFolder)
# Path to the log file of the script. 
logPath=os.path.join(mainFolder,'job.log')
# Generate log file and print header
printLogHeader()
# Initiate count of jobs finished
count=0
# Start jobs
for n in range(n_jobs):
    molName=job_name+'_'+str(n)
    # Check if molName requires special config file
    if cas_number in noautoz:
        configFile=os.path.join(mainFolder,
                                '..',
                                'Python',
                                'lib',
                                '_config',
                                nwchemConfig+'_noautoz.config')
        print(f'\nUsing noautoz config file: {configFile}\n')
    elif cas_number in iodine:
        configFile=os.path.join(mainFolder,
                                '..',
                                'Python',
                                'lib',
                                '_config',
                                nwchemConfig+'_Iodine.config')
        print(f'\nUsing Iodine config file: {configFile}\n')
    else:
        configFile=os.path.join(mainFolder,
                                '..',
                                'Python',
                                'lib',
                                '_config',
                                nwchemConfig+'.config')
        print(f'\nUsing default config file: {configFile}\n')
    # Call generateSP
    __,t,e=call_generateSP([molName,identifier],configFile)
    # Update count
    count+=1
    # Write information to log file
    with open(logPath,'a') as logFile:
        if e: logFile.write('\n'+molName+' finished with errors.\n')
        else: logFile.write('\n'+molName+' finished successfully.\n')
        logFile.write('Wall clock time for this job: '+str(t)+' s\n')
        logFile.write('So far, '+str(count)+'/'+str(n_jobs)+' jobs finished.\n')