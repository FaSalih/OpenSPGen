#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================

# General
import os
import time
import sys
import csv
import traceback
import pandas as pd
import numpy as np
import argparse, sys

# Local
from lib import spGenerator as sp

# =============================================================================
# Configuration (user-provided options)
# =============================================================================

# Parse user arguments
parser=argparse.ArgumentParser()

parser.add_argument("--idtype",required=True,  help="Molecule identifier type. Options: SMILES, CAS-Number, InChI, InChIKey, or mol2 (Not case sensitive, but must include separators like `-`). This argument is required.")
parser.add_argument("--id", required=True, help="Molecule identifier. This argument is required.")
parser.add_argument("--charge", help="Molecule charge. Default is None and will be calculated later on using `rdkit.Chem.rdmolops`.")
parser.add_argument("--initialxyz", help="Path to initial xyz file for NWChem geometry optimization, if desired. Otherwise, use 'Random' or 'None' for a random conformer.")
parser.add_argument("--preoptimize", help="Pre-optimize the molecule using a standard forcefield (MMFF94). Options: True or False. Only available if a `mol2` idtype is provided.")
parser.add_argument("--name", help="Tail for the job name. Default is `UNK`.")
parser.add_argument("--nslots", help="Number of cores/threads to use for NWChem calculations. Default is 4.")
parser.add_argument("--njobs", help="Number of repeat jobs to be run. Default is 1.")
parser.add_argument("--noautoz", help="NWChem setting to disable use of internal coordinates. Default is False.")
parser.add_argument("--iodine", help="The molecule contains an iodine atom. Default is False.")

args=parser.parse_args()

# =============================================================================
# Configuration (fixed options)
# =============================================================================

# NWChem Config file base name
nwchemConfig='COSMO_HF_SVP'
# Do COSMO? (= calculate sigma profile, not just sigma surface)
doCOSMO=True
# Molecule requires noautoz config file
noautoz=False
# Molecule requires iodine config file
iodine=False
# Other spGenerator.py options:
cleanOutput=True
removeNWOutput=False
generateFinalXYZ=True
generateOutputSummary=True
avgRadius=None
sigmaBins=[-0.250,0.250,0.001]
noautoz=False
iodine=False


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
    # if initialXYZ_ is provided, adjust path:
    if initialXYZ_ not in [None, 'Random']: # does not wok for provided xyz
        initialXYZ_=os.path.join('..',initialXYZ_)
    try:
        warning=sp.generateSP(entry[1],jobFolder,np_NWChem,configFile,
                              identifierType=identifierType,
                              charge=charge,
                              initialXYZ=initialXYZ_,
                              randomSeed=randomSeeds[n],
                              cleanOutput=cleanOutput,
                              removeNWOutput=removeNWOutput,
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
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s\n" % (trace[0], trace[1], trace[2], trace[3]))

            print("Exception type : %s " % ex_type.__name__)
            print("Exception message : %s" %ex_value)
            print("Stack trace : %s" %stack_trace)   
            logFile.write('\nException type : %s ' % ex_type.__name__)
            logFile.write('\nException message : %s' %ex_value)
            logFile.write('\nStack trace : %s' %stack_trace)
        errorOcurred=True
    # Return to parent directory
    os.chdir(mainFolder)
    # Get elapsed time
    t=round(time.time()-t1,2)
    # Output
    return entry,t,errorOcurred

def printLogHeader(logPath):
    """
    printLogHeader() prints the details of the parallel job to the log file.
    To avoid unecessarily heavy code, this function accesses variables outside
    of its scope, which are not global and are not changed inside it.

    Arguments
    ---------
    logPath : string
        Path to the log file.
    Returns
    -------
    None.

    """
    # Create log file
    with open(logPath,'a') as logFile:
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

def parseUserArgs(userArgs):
    """
    parseUserArgs() parses the user arguments, checks input validity, and defines variables from user input.

    Arguments:
    -------
    userArgs : dictionary
        Dictionary containing the user arguments.

    Returns:
    -------
    identifier : string
        Molecule identifier.
    identifierType : string
        Molecule identifier type.
    charge : float
        Molecule charge.
    initialXYZ : string
        Path to initial xyz file, if desired. Otherwise, use 'Random' or 'None' for a random conformer.
    preOptimize : boolean
        Pre-optimize the molecule using a standard forcefield (MMFF).
    job_name : string
        Full for the job name.
    np_NWChem : int
        Number of cores/threads to use for NWChem calculations.
    logPath : string
        Path to the log file.
    mainFolder : string
        Path to the main folder the job is being run from. Contains the Python folder and the current job folder.
    """ 
    # Define defaults
    default_options={
        'idtype': 'SMILES',
        'id': None,
        'charge': None,
        'initialxyz': None,
        'preoptimize': False,
        'name': 'UNK',
        'nslots': 4,
        'njobs': 1,
        'noautoz': False,
        'iodine': False
    }

    # Check if user provided idtype is valid
    if userArgs.idtype is not None:
        if userArgs.idtype.lower() not in ['smiles', 'cas-number', 'inchi', 'inchikey', 'mol2']:
            # Terminate with an error
            print(f'\n\tInput error:')
            print(f'\n\t\tThe value provided for the "--idtype" argument is invalid. Please provide one of the following options: SMILES, CAS-Number, InChI, InChIKey, or mol2.')
            sys.exit(1)

    # Set job_name_tail
    if userArgs.name is not None:
        job_name_tail=userArgs.name
    else:
        job_name_tail=default_options['name']

    # Set nslots (number of available cores)
    if userArgs.nslots is not None:
        nslots=userArgs.nslots
        if int(nslots)<1:
            # Terminate with an error
            print(f'\n\tInput error:')
            print(f'\n\t\tThe value provided for the "--nslots" argument is invalid. Please provide a positive integer.')
            sys.exit(1)
    else:
        nslots=default_options['nslots']
    np_NWChem=nslots

    # Read user-defined charge
    if userArgs.charge is not None:
        charge=userArgs.charge
    else:
        charge=default_options['charge']     

    # Read user-selected NWChem configuration options
    if userArgs.noautoz is not None:
        noautoz=userArgs.noautoz
        # Check if provided value is valid
        if noautoz.lower() not in ['true', 'false']:
            # Terminate with an error
            print(f'\n\tInput error:')
            print(f'\n\t\tThe value provided for the "--noautoz" argument is invalid. Please provide either "True" or "False".')
            sys.exit(1)
    else:
        noautoz=default_options['noautoz']
    if userArgs.iodine is not None:
        iodine=userArgs.iodine
        # Check if provided value is valid
        if iodine.lower() not in ['true', 'false']:
            # Terminate with an error
            print(f'\n\tInput error:')
            print(f'\n\t\tThe value provided for the "--iodine" argument is invalid. Please provide either "True" or "False".')
            sys.exit(1)
    else:
        iodine=default_options['iodine']

    # Read user-defined number of jobs
    if userArgs.njobs is not None:
        nJobs=int(userArgs.njobs)
    else:
        nJobs=default_options['njobs']        
    
    # Random seeds for initial conformer generation
    randomSeeds=[42+nJob for nJob in range(nJobs)]

    # Convert initialxyz from string if needed
    if userArgs.initialxyz.upper() in ['NONE', None]:
        initialXYZ=None
    else:
        initialXYZ=userArgs.initialxyz

    # Specify full job name
    if initialXYZ is None:
        job_name=f'SP-NoInitXYZ-Mol_{job_name_tail}'
    elif initialXYZ.upper() in ['RANDOM', 'RAND']:
        job_name=f'SP-RandInitXYZ-Mol_{job_name_tail}'
    else:
        job_name=f'SP-GivenInitXYZ-Mol_{job_name_tail}'

    # Path to the main folder
    mainFolder=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..',
                            job_name)
    # Make main folder
    if not os.path.isdir(mainFolder): os.makedirs(mainFolder)

    # Path to the log file of the script. 
    logPath=os.path.join(mainFolder,'job.log')
    
    # Process user arguments and replace with defaults if not provided
    with open(logPath,'a') as logFile:  
        logFile.write('\nProcessing user arguments...')

    # Check validity of input geometry and identifier options
    if userArgs.idtype.lower()=="mol2":
        with open(logPath,'a') as logFile:
            logFile.write(f'\n\tUsing provided initial geometry in mol2 file: {userArgs.id}')
        identifierType=userArgs.idtype
        identifier=userArgs.id
        # Check if pre-optimization is desired
        if userArgs.preoptimize is not None:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry is set by the user to: {userArgs.preoptimize}')
            preOptimize=userArgs.preoptimize
        else:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry was not set. Default for a mol2 input file is: True')
            preOptimize=True
    elif userArgs.initialxyz.upper() not in [None, 'NONE', 'RANDOM', 'RAND']:
        with open(logPath,'a') as logFile:
            logFile.write(f'\n\tUsing provided initial xyz file: {userArgs.initialxyz}')
        initialXYZ=userArgs.initialxyz

        # State pre-optimization availability
        if userArgs.preoptimize is not None:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry is set by the user to: {userArgs.preoptimize}. This option is only available for mol2 idtypes.')
            preOptimize=False
            identifier=userArgs.id
        else:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry was not set. Default is: {default_options["preoptimize"]}')
            preOptimize=default_options['preoptimize']
            # Check if identifier was provided
            if userArgs.id is None:
                with open(logPath,'a') as logFile:
                    logFile.write(f'\n\tNo identifier provided and none is needed. Will use assume {default_options["idtype"]} for the "--idtype" and {default_options["id"]} in place of the "--id" argument.')
                identifierType=default_options['idtype']
                identifier=default_options['id']
            else:
                with open(logPath,'a') as logFile:
                    logFile.write(f'\n\tIdentifier information is provided but is not needed.')
                identifierType=userArgs.idtype
                identifier=userArgs.id

    elif userArgs.initialxyz in ['Random', 'Rand']:
        print(f'\n\tUsing random initial Geometry.')
        initialXYZ='Random'
        # Check if identifier was provided
        if userArgs.id is None:
            # Terminate with an error
            with open(logPath,'a') as logFile:
                    logFile.write('\n\tInput error:')
                    logFile.write(f'\n\t\tRandom initial geometry requires providing an "--id" argument with a SMILES, CAS-Number or InChIKey identifier.')
            sys.exit(1)
        else:
            # Set pre-optimization to true
            with open(logPath,'a') as logFile:
                    logFile.write('\n\tPre-optimizaion using MMFF of random generated geometry is set to: True')
            preOptimize=True
            # Save identifier
            identifier=userArgs.id
    else:
        with open(logPath,'a') as logFile:
                    logFile.write(f'\n\tNo initial geometry provided. Will use a random conformer.')
        initialXYZ=None
        # Check if identifier was provided
        if userArgs.id is None:
            # Terminate with an error
            with open(logPath,'a') as logFile:
                    logFile.write('\n\tInput error:')
                    logFile.write(f'\n\t\tRandom initial geometry requires providing an "--id" argument with a SMILES, CAS-Number InChI or InChIKey identifier.')
            sys.exit(1)
        else:
            # Check that identifier type was provided
            if userArgs.idtype is None:
                # Set to default
                with open(logPath,'a') as logFile:
                    logFile.write(f"\n\tIdentifier type was not provided. Defaulting to: {default_options['idtype']}")
                identifierType=default_options['idtype']
                # Save identifier
                identifier=userArgs.id
            else:
                with open(logPath,'a') as logFile:
                    logFile.write(f'\n\tIdentifier type was provided: {userArgs.idtype}')
                identifierType=userArgs.idtype
            # Save identifier
            identifier=userArgs.id
            # Set pre-optimization to true
            with open(logPath,'a') as logFile:
                    logFile.write('\n\tPre-optimizaion using MMFF of a random geometry is set to: True')
            preOptimize=True

    # Create log file
    with open(logPath,'a') as logFile:
        logFile.write('\n\nInitializing serial task...\n')
        logFile.write('\tMain folder: '+mainFolder+'\n')
        # logFile.write('\tMolecule list: '+identifierListPath+'\n')
        logFile.write('\tNumber of threads per job: '+str(np_NWChem)+'\n')
        logFile.write('\tNWChem configuration file: '+nwchemConfig+'\n')
        logFile.write('\tDo COSMO: '+str(doCOSMO)+'\n')
        if doCOSMO:
            logFile.write('\tAveraging radius: '+str(avgRadius)+'\n')
            logFile.write('\tSigma bins: '+str(sigmaBins)+'\n')
        logFile.write('Initialization complete.\n')

    # return user-defined variables
    return (
        identifier, identifierType, charge, initialXYZ, 
        preOptimize, job_name, np_NWChem, logPath, 
        mainFolder, nJobs, randomSeeds, noautoz, iodine
        )


# =============================================================================
# Main Script
# =============================================================================
# Parse user arguments
(
    identifier, identifierType, charge, initialXYZ, 
    preOptimize, job_name, np_NWChem, logPath, 
    mainFolder, nJobs, randomSeeds, noautoz, iodine
 )=parseUserArgs(args) 
# Initiate count of jobs finished
count=0
# Start jobs
for n in range(nJobs):
    molName=job_name+'_'+str(n)
    # Check if molName requires special config file
    if noautoz:
        configFile=os.path.join(mainFolder,
                                '..',
                                'Python',
                                'lib',
                                '_config',
                                nwchemConfig+'_noautoz.config')
        print(f'\nUsing noautoz config file: {configFile}\n')
    elif iodine:
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
        logFile.write('So far, '+str(count)+'/'+str(nJobs)+' jobs finished.\n')
