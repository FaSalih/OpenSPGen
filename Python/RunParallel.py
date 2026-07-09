#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================

# General
import os
import time
import sys
import traceback
import argparse, sys

# Local
import os.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from lib import spGenerator as sp

# =============================================================================
# Configuration (user-provided options)
# =============================================================================

# Parse user arguments
parser=argparse.ArgumentParser()

parser.add_argument("--idtype",required=True,  help="Molecule identifier type. Options: SMILES, CAS-Number, InChI, InChIKey, mol2, or xyz (Not case sensitive, but must include separators like `-`). This argument is required.")
parser.add_argument("--id", required=True, help="Molecule identifier. This argument is required.")
parser.add_argument("--charge", help="Molecule charge. Default is None and will be calculated later on using `rdkit.Chem.rdmolops`.")
parser.add_argument("--initialxyz", help="Path to an initial xyz file for NWChem geometry optimization. If omitted, a random-seeded geometry is used.")
parser.add_argument("--preoptimize", help="Pre-optimize the molecule using a standard forcefield (MMFF94). Options: True or False. Only available if a `mol2` idtype is provided.")
parser.add_argument("--confid", required=True, help="Conformer ID or job number. Maps the provided index to a random seed for structure intialization. Indexing starts at 0 corresponding to a random seed of 42.")
parser.add_argument("--name", required=True, help="Tail for the job name. Job names are required so that different conformer jobs of the same molecule are saved under the same molecule folder.")
parser.add_argument("--nslots", help="Number of cores/threads to use for NWChem calculations. Default is 4.")
parser.add_argument("--noautoz", help="NWChem setting to disable use of internal coordinates. Default is False.")
parser.add_argument("--iodine", help="The molecule contains an iodine atom. Default is False.")

args=parser.parse_args()

# =============================================================================
# Configuration (fixed options)
# =============================================================================

# NWChem Config file base name - full config file path is f"Python/lib/_config/{nwchemConfig}.config"
nwchemConfig='COSMO_HF_SVP'
# Do COSMO? (= calculate sigma profile, not just sigma surface)
doCOSMO=True
# Other spGenerator.py options:
cleanOutput=True        # delete auxiliary NWChem files (e.g. job_name.movecs, job_name.drv.hess, job_name.db)
removeNWOutput=True     # delete NWChem output file
generateFinalXYZ=True   # generate xyz file for final optimized geometry
generateOutputSummary=True     # generate output summary file (includes energies from last optimization step)
avgRadius=None                  # averaging radius for converting sigma surface to sigma profile
sigmaBins=[-0.250,0.250,0.001]  # charge density bins. The range here is larger than needed to prevent jobs from crashing

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
    # if initialXYZ_ is provided, convert relative to absolute path:
    if initialXYZ_ not in [None, 'Random']:
        if not os.path.isabs(initialXYZ_):
            initialXYZ_ = os.path.abspath(initialXYZ_)
    try:
        warning=sp.generateSP(entry[1],jobFolder,np_NWChem,configFile,logPath,
                              identifierType=identifierType,
                              charge=charge,
                              initialXYZ=initialXYZ_,
                              randomSeed=randomSeed,
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

            # Extract unformatted stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace into a readable multiline string
            stack_lines = []
            for trace in trace_back:
                # trace fields: (filename, lineno, name, line)
                filename = trace[0]
                lineno = trace[1]
                funcname = trace[2]
                message = trace[3] if trace[3] is not None else ''
                stack_lines.append(f"  File: {filename}, line {lineno}, in {funcname}\n    {message}\n")

            stack_trace_str = "".join(stack_lines)

            # Print to stdout in a readable format
            print(f"Exception type : {ex_type.__name__}")
            print(f"Exception message : {ex_value}")
            print("Stack trace:\n" + stack_trace_str)

            # Write readable stack trace to log
            logFile.write('\nException type : %s ' % ex_type.__name__)
            logFile.write('\nException message : %s' % ex_value)
            logFile.write('\nStack trace:\n%s' % stack_trace_str)
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
        Path to an initial xyz file, if provided. Otherwise, a random-seeded geometry is used.
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
        'initialxyz': 'RANDOM',
        'preoptimize': False,
        'name': None,
        'nslots': 4,
        'confid': 0,
        'noautoz': False,
        'iodine': False
    }

    # Check if user provided idtype is valid
    if userArgs.idtype is not None:
        if userArgs.idtype.lower() not in ['smiles', 'cas-number', 'inchi', 'inchikey', 'mol2', 'xyz']:
            # Terminate with an error
            print(f'\n\tInput error:')
            print(f'\n\t\tThe value provided for the "--idtype" argument is invalid. Please provide one of the following options: SMILES, CAS-Number, InChI, InChIKey, mol2, or xyz.')
            sys.exit(1)

    # Set job_name_tail
    job_name_tail=userArgs.name

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
            iodine = True if iodine.lower()=='true' else False
    else:
        iodine=default_options['iodine']

    # Read user-defined number of jobs
    if userArgs.confid is not None:
        confid=int(userArgs.confid)
    else:
        confid=default_options['confid']        
    
    # Random seeds for initial conformer generation
    randomSeed=42+confid

    # Determine whether a supplied initial geometry or a random-seeded geometry should be used
    if userArgs.initialxyz is None or userArgs.initialxyz.upper() in ['RANDOM', 'RAND']:
        initialXYZ='Random'
        useRandomGeometry=True
    else:
        initialXYZ=userArgs.initialxyz
        useRandomGeometry=False

    # Specify full job name
    if useRandomGeometry:
        job_name=f'SP-RandInitXYZ-Mol_{job_name_tail}'
    else:
        job_name=f'SP-GivenInitXYZ-Mol_{job_name_tail}'

    # Path to the OpenSPGen scripts and libraries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to config files
    config_dir = os.path.join(script_dir, 'lib', '_config')
    
    # Main job folder will be inside the current working directory
    mainFolder = os.path.join(os.getcwd(), job_name)
    # Make main job folder, if it doesn't exist alread
    os.makedirs(mainFolder, exist_ok=True)

    # Path to the log file of the script. 
    logPath=os.path.join(mainFolder,f'job_conf_{confid}.log')
    
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
    elif useRandomGeometry:
        print(f'\n\tUsing random initial geometry with random seed: {randomSeed}.')
        if userArgs.id is None:
            with open(logPath,'a') as logFile:
                logFile.write('\n\tInput error:')
                logFile.write(f'\n\t\tRandom initial geometry requires providing an "--id" argument with a SMILES, CAS-Number, InChI, or InChIKey identifier.')
            sys.exit(1)

        with open(logPath,'a') as logFile:
            logFile.write('\n\tPre-optimizaion using MMFF of a random generated geometry is set to: True')
        preOptimize=True
        identifierType=userArgs.idtype if userArgs.idtype is not None else default_options['idtype']
        identifier=userArgs.id
    else:
        with open(logPath,'a') as logFile:
            logFile.write(f'\n\tUsing provided initial xyz file: {initialXYZ}')

        if userArgs.preoptimize is not None:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry is set by the user to: {userArgs.preoptimize}.')
            preOptimize=userArgs.preoptimize
        else:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tPre-optimizaion using MMFF of provided geometry was not set. Default is: {default_options["preoptimize"]}')
            preOptimize=default_options['preoptimize']

        if userArgs.id is None:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tNo identifier provided. Either an identifier or charge are needed for the supplied initial xyz geometry.')
            # Check if charge is provided
            if charge is None:
                with open(logPath,'a') as logFile:
                    logFile.write(f'\n\tNo identifier or charge are needed for the supplied initial xyz geometry.')
                    logFile.write('\n\tInput error:')
                    logFile.write(f'\n\t\tUsing a provided initial xyz geometry requires providing either a charge through the "--charge" argument '\
                                  + 'or an "--id" argument with a SMILES, CAS-Number, InChI, or InChIKey identifier to allow calculating charge.')
                sys.exit(1)
            identifierType=default_options['idtype']
            identifier=default_options['id']
        else:
            with open(logPath,'a') as logFile:
                logFile.write(f'\n\tIdentifier information is provided but is not needed.')
            identifierType=userArgs.idtype
            identifier=userArgs.id

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
        mainFolder, confid, randomSeed, noautoz, iodine,
        config_dir
        )

# =============================================================================
# Main Script
# =============================================================================
# Parse user arguments
(
    identifier, identifierType, charge, initialXYZ, 
    preOptimize, job_name, np_NWChem, logPath, 
    mainFolder, confid, randomSeed, noautoz, iodine,
    config_dir
 )=parseUserArgs(args) 
# Initiate count of jobs finished
count=0
# Start jobs
molName=job_name+'_'+str(confid)
# Check if molName requires special config file
if noautoz:
    configFile=os.path.join(config_dir, nwchemConfig+'_noautoz.config')
    print(f'\nUsing noautoz config file: {configFile}\n')
elif iodine:
    configFile=os.path.join(config_dir, nwchemConfig+'_Iodine.config')
    print(f'\nUsing Iodine config file: {configFile}\n')
else:
    configFile=os.path.join(config_dir, nwchemConfig+'.config')
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
