# -*- coding: utf-8 -*-
"""
NWChem_Wrapper is a wrapper for NWChem. Its main functions are to build the
NWChem input script, run NWChem, and retrieve results from the output files.

Sections
    . Imports
    
    . Main Functions
        . buildInputFile()
        . runNWChem()
        . readCOSMO()
        . readOutput()
        . checkConvergence()
        . generateFinalXYZ()
        . generateLastStep()
    
    . Auxiliary Functions
        . copyConfig()
        . findLastOccurrence()
        . findAllOccurrences()
        . goToLine()
        . findNextOccurrence()

Last edit: 2022-07-11
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import subprocess

# =============================================================================
# Main Functions
# =============================================================================

def buildInputFile(inputPath,configPath,xyzPath,name,charge):
    """
    buildInputFile() builds an NWChem input file based on the inputs provided
    by the user and the .config file present in lib/_config.

    Parameters
    ----------
    inputPath : string
        Path to the input file being created.
    configPath : string
        Path to the nwchem configuration file. See /path/to/lib/config.
    xyzPath : string
        Path to the the xyz file containing the initial geometry of the
        molecule.
    name : string
        Name of the molecule/job.
    charge : int
        Total charge of the molecule.

    Returns
    -------
    None.

    """
    # Open input file
    with open(inputPath,'w') as inputFile:
        # Write info header
        inputFile.write('#----------------------------------------------\n'
                        +'# Automatically generated with NWChem_Wrapper\n'
                        +'#----------------------------------------------\n\n')
        # Write job name
        inputFile.write('start '+name+'\n\n')
        # Write configuration header
        inputFile.write('#----------------------------------------------\n'
                        +'# System Configuration\n'
                        +'#----------------------------------------------\n\n')
        # Write charge of the molecule/ion
        inputFile.write('charge '
                        +str(charge)
                        +' # Charge of the molecule/ion\n')
        # Copy configuration file
        copyConfig(inputFile,configPath)
    # Output
    return None
        
def runNWChem(inputPath,jobFolder,np,runCommand=None):
    """
    runNWChem() runs NWChem. By default, the run command is:
        mpirun -np XX nwchem YYY
    where XX is the number of threads to use and YYY is the name/path of the
    input file. The user can provide a custom run command.

    Parameters
    ----------
    inputPath : string
        Path to the NWChem input file.
    jobFolder : string
        Path of the job folder (i.e. where output files are stored).
    np : int
        Number of threads to be used by NWChem.
    runCommand : string, optional
        If runCommand is not None, the function calls NWChem using:
            subprocess.call(runCommand)
        The default is None.

    Returns
    -------
    None.

    """
    # Change working directory to jobFolder
    os.chdir(jobFolder)
    # Open output file
    with open('output.nw','w') as outputFile:
        # If user did not provide a runCommand, use default
        if runCommand is None:
            subprocess.call('mpirun -np '+str(np)+' nwchem '+inputPath,
                            shell=True,
                            stdout=outputFile)
        else: # Use user-provided runCommand
            subprocess.call(runCommand,shell=True,stdout=outputFile)
    # Output
    return None
    
def readCOSMO(cosmoPath):
    """
    readCOSMO() reads NWChem output files of the type "cosmo.xyz" and returns
    the coordinates and charges of all segments in the file.

    Parameters
    ----------
    cosmoPath : string
        Path to the "cosmo.xyz" file of interest.

    Returns
    -------
    segmentCoordinates : list of list of floats
        List where each entry corresponds to a list with the coordinates
        (x,y,z) of a segment.
    segmentCharges : list of floats
        List where each entry corresponds to the charge of a segment.

    """
    # Initialize coordinates and charge containers
    segmentCoordinates=[]
    segmentCharges=[]
    # Open input file
    with open(cosmoPath,'r') as file:
        # Get total number of segments from first line
        nSegments=int(file.readline().strip())
        # Skip empty line
        file.readline()
        # Loop with nSegments iterations
        for __ in range(nSegments):
            # Get coordinates line and split
            coordLine=file.readline().split() # "Bq x y z"
            # Append coordinates to container
            segmentCoordinates.append([float(coordLine[1]),
                                       float(coordLine[2]),
                                       float(coordLine[3])])
            # Get charge line
            chargeLine=file.readline().split() # Single value
            # Append symmetric of charge to container
            segmentCharges.append(-float(chargeLine[0]))
    # Output
    return segmentCoordinates,segmentCharges

def readOutput(outputPath,doCOSMO=True):
    """
    readOutput() reads an NWChem output file and retrieves the following
    information from the last optimization step:
        . The total area of the molecule, as calculated by NWChem
        . A list with the area of each segment
        . The final coordinates of all the atoms
    
    Parameters
    ----------
    outputPath : string
        Path to the output file of NWChem.
    doCOSMO : bool, optional
        Whether to read COSMO-related information.
        The default is true.

    Returns
    -------
    surfaceArea : float or None
        Total surface area of the molecule (Ang^2).
    segmentAreas : list of floats or None
        List with the surface area of each segment (a.u.^2).
    atomCoords : list of lists of floats
        List containing information about atom positions. Each entry
        corresponds to a different atom and contains:
            Element (string)
            X coordinate (float) (Ang)
            Y coordinate (float) (Ang)
            Z coordinate (float) (Ang)
    segAtoms : list of ints
        List containing the atom indices that each segment belongs to.

    """
    # Open output file
    with open(outputPath,'r') as file:
        if doCOSMO:
            # Find last occurrence of "-cosmo- solvent"
            lastOccurrenceLine=findLastOccurrence(file,['-cosmo-','solvent'])
            # Go to line of last occurrence
            goToLine(file,lastOccurrenceLine)
            # Retrieve number of segments per atom
            findNextOccurrence(file,'nspa,') # find table header
            file.readline() # skip separator
            # Initialize number of elements per atom
            numSegments=[]
            while True:
                # Read line
                lineSplit=file.readline().split()
                nspa=lineSplit[2][:-1]
                # Check if table has ended
                if lineSplit[0] == 'number': break
                numSegments.append(nspa)
            # Populate array with segment assignments
            segAtoms = [[i+1]*int(numSegments[i]) for i in range(len(numSegments))]
            flat_list = lambda main_list: [item for sublist in main_list for item in sublist]
            segAtoms = flat_list(segAtoms)
            # Retrieve total number of segments
            nSeg=int(lineSplit[-1])
            # Retrieve total surface area
            lineSplit=file.readline().split()
            surfaceArea=float(lineSplit[3])
            # Find table with the area of each segment
            findNextOccurrence(file,'G(cav/disp)')  # find table header
            # Initialize list of areas
            segmentAreas=[]
            # Find first entry of the table
            lineSplit=findNextOccurrence(file,'1')
            segmentAreas.append(float(lineSplit[1]))
            # Read remaining areas
            for __ in range(nSeg-1):
                lineSplit=file.readline().split()
                segmentAreas.append(float(lineSplit[1]))
            # Find table of final geometry
            findNextOccurrence(file,'1.889725989')
        else:
            # Find last occurrence of "1.889725989"
            lastOccurrenceLine=findLastOccurrence(file,['Output',
                                                        'coordinates',
                                                        'in',
                                                        'angstroms',
                                                        '(scale',
                                                        'by',
                                                        '1.889725989',
                                                        'to',
                                                        'convert',
                                                        'to',
                                                        'a.u.)'])
            # Go to line of last occurrence
            goToLine(file,lastOccurrenceLine)
            # Set COSMO-related outputs to None
            surfaceArea=None
            segmentAreas=None
        # Skip three lines
        for __ in range(3): file.readline()
        # Retrieve final coordinates
        atomCoords=[] # Initialize coordinates list
        while True:
            # Read line
            lineSplit=file.readline().split()
            # Check if coordinates table is over
            if lineSplit==[]: break
            # Append element and coordinates
            atomCoords.append([lineSplit[1],
                               float(lineSplit[3]),
                               float(lineSplit[4]),
                               float(lineSplit[5])])
    # Output
    return surfaceArea,segmentAreas,atomCoords,segAtoms

def checkConvergence(outputPath):
    ''' 
    checkConvergence() checks if the optimization converged by reading the 
    output file of NWChem.
    Arguments:
        outputPath : string
            Path to the output file of NWChem.
    Returns:
        converged : int
            1 if the optimization converged in both vacuum and the solvation medium
            0 if the optimization converged only in vacuum
            -1 if the optimization did not converge in neither vacuum nor the solvation medium
    '''
    # Open output file
    with open(outputPath,'r') as file:
        # Find last occurrence of "Optimization converged"
        lastOptimLine=findLastOccurrence(file,['Optimization','converged'])
        # Check if any "optimization converged" lines were found
        if lastOptimLine is not None:
            # Find last occurrence of "-cosmo- solvent"
            lastCosmoLine=findLastOccurrence(file,['-cosmo-','solvent'])
            # Check if any "cosmo" lines were found
            if lastCosmoLine is None:
                converged=0
            else:
                # Check that the last successful optimization is after cosmo ended
                if lastOptimLine > lastCosmoLine:
                    converged=1
                else:
                    converged=0
        else:
            converged=-1
    # Output
    return converged

def generateFinalXYZ(atomCoordsList,xyzPath):
    """
    generateFinalXYZ() generates an XYZ file using the information contained
    in coords.

    Parameters
    ----------
    atomCoordsList : list of lists
        List containing information about atom positions. Each entry
        corresponds to a different atom and contains:
            Element (string)
            X coordinate (float)
            Y coordinate (float)
            Z coordinate (float)
    xyzPath : string
        Path where the XYZ file is saved.

    Returns
    -------
    None.

    """
    # Open xyz file
    with open(xyzPath,'w') as xyzFile:
        # Write first line with total number of atoms
        xyzFile.write(str(len(atomCoordsList))+'\n\n')
        # Iterate over coords entries
        for line in atomCoordsList:
            # Write atom line (el x y z)
            xyzFile.write(line[0]
                          +'   '+str(line[1])
                          +'   '+str(line[2])
                          +'   '+str(line[3])+'\n')
    # Output
    return None
    
def generateLastStep(outputPath,summaryPath):
    """
    generateLastStep() generates a summary output file containing only the last
    step of the full NWChem output file.

    Parameters
    ----------
    outputPath : string
        Path of the output file.
    summaryPath : string
        Path of the summary file.

    Returns
    -------
    None.

    """
    # Open output file
    with open(outputPath,'r') as outputFile:
        # Find last occurrence of "-cosmo- solvent"
        lastOccurrenceLine=findLastOccurrence(outputFile,
                                              ['Optimization','converged'])
        # Go to line of last occurrence
        goToLine(outputFile,lastOccurrenceLine)
        # Open summary file
        with open(summaryPath,'w') as summaryFile:
            # Copy everything to summary file, until outputFile reaches 
            # "...Task  times  cpu..."
            while True:
                # Read next line of output file
                line=outputFile.readline()
                # Check if line is "...Task  times  cpu..."
                if line.split()[0:3]==['Task','times','cpu:']: break
                # Copy line to summary file
                summaryFile.write(line)
    # Output
    return None    

# =============================================================================
# Auxiliary Functions
# =============================================================================

def copyConfig(inputFile,configPath):
    """
    copyConfig() reads the nwchem configuration file and copies everything
    below the line "-------78963b1b48f356a19a3bdc8650728784-------"

    Parameters
    ----------
    inputFile : _io.TextIOWrapper object
        NWChem input file.
    configPath : string
        Path to the nwchem configuration file. See /path/to/lib/config.

    Returns
    -------
    None.

    """
    
    # Get default config path
    # configPath=os.path.join(os.path.dirname(__file__),
    #                         '_config',
    #                         'nwchem.config')
    
    # Open config file
    with open(configPath,'r') as configFile:
        # Find 78963b1b48f356a19a3bdc8650728784
        findNextOccurrence(configFile,
                           '-------78963b1b48f356a19a3bdc8650728784-------')
        # Copy remaining file
        for line in configFile: inputFile.write(line)
    # Output
    return None

def findLastOccurrence(file,targetLineSplit):
    """
    findLastOccurrence() finds the line number of the last occurrence of a line
    in a file.

    Parameters
    ----------
    file : _io.TextIOWrapper object
        File of interest.
    targetLineSplit : list of strings
        Desired target line, inputted as a list of splits (ex. obtained using
        line.split()).
    Returns
    -------
    lastOccurrenceLine : int
        Number of the line where targetLineSplit last occurs.

    """
    # Rewind file
    file.seek(0)
    # Initiate line counter
    lineNum=0
    # Initiate variables
    lastOccurrenceLine=None
    # Read file line by line
    for line in file:
        # Split line
        lineSplit=line.split()
        # Check if split is the target split
        if lineSplit==targetLineSplit:
            # Update line number of last occurrence
            lastOccurrenceLine=lineNum
        # Update line number
        lineNum+=1
    # Output
    return lastOccurrenceLine

def findAllOccurrences(file,targetLineSplit):     
    """
    findAllOccurrences() finds the line number of all occurrences of a line
    in a file.

    Parameters
    ----------
    file : _io.TextIOWrapper object
        File of interest.
    targetLineSplit : list of strings
        Desired target line, inputted as a list of splits (ex. obtained using
        line.split()).
    Returns
    -------
    findAllOccurrences : list of int
        Number of the lines where targetLineSplit occurs.

    """
    # Rewind file
    file.seek(0)
    # Initiate line counter
    lineNum=0
    # Initialize list of occurrences
    allOccurrenceLines=[]
    # Read file line by line
    for line in file:
        # Split line
        lineSplit=line.split()
        # Check if split is the target split
        if lineSplit==targetLineSplit:
            # Update line number of last occurrence
            allOccurrenceLines.append(lineNum)
        # Update line number
        lineNum+=1
    # Output
    return allOccurrenceLines

def goToLine(file,lineNumber):
    """
    goToLine() places the pointer of file at the line with number "lineNumber"

    Parameters
    ----------
    file : _io.TextIOWrapper object
        File of interest.
    lineNumber : int
        Number of the line of interest.

    Returns
    -------
    None.

    """
    # Rewind file
    file.seek(0)
    # Initiate loop
    for __ in range(lineNumber+1):
        file.readline()
    # Output
    return None

def findNextOccurrence(file,fragment):
    """
    findNextOccurrence() brings the file pointer to the next line where
    "fragment" is found inside a line split list.

    Parameters
    ----------
    file : _io.TextIOWrapper object
        File of interest.
    fragment : string
        Fragment to find.

    Returns
    -------
    lineSplit : list of strings
        Split of the line found.

    """
    # Initiate loop
    for line in file:
        # Read line and split
        lineSplit=line.split()
        # If lineSplit contains fragment, break
        if fragment in lineSplit: break
    # Output
    return lineSplit
