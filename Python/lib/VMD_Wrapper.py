# -*- coding: utf-8 -*-
"""
[NOT FULLY IMPLEMENTED]
VMD_Wrapper.py is a wrapper for VMD.
# Might not work outside of Windows; change in the future...

Sections
    . Imports
    
    . Main Function
        . viewSigmaSurface()
        . xyz2bmp()
        . openVMD()
        . viewFinalGeometries()
    
    . Auxiliary Functions

Last edit: 2024-11-19
Author: Dinis Abranches, Fathya Salih
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import secrets
import subprocess
import shutil

# Specific
import numpy
from tqdm import tqdm

# Local
from . import spGenerator as sp

# =============================================================================
# Main Functions
# =============================================================================

def viewSigmaSurface(surfaceCSV,geometryXYZ,vmdPath,avgRadius=0.5,bins=[-0.100,0.100,0.001]):
    
    
    # Read information from sigmaSurface file
    sigmaMatrix=numpy.genfromtxt(surfaceCSV,delimiter=',')
    # Perform averaging algorithm, if requested
    if avgRadius is not None:
        sigmaMatrix=sp.averagingAlgorithm(sigmaMatrix,avgRadius)
    # Sort sigmaMatrix by charge density
    sigmaMatrix=sigmaMatrix[sigmaMatrix[:,5].argsort()]
    # Generate sigma bins (location of bin center, sigma vector)
    sigma=numpy.arange(bins[0],bins[1]+bins[2],bins[2])
    # Get number of bins
    nBins=len(sigma)
    # There are 32 color IDs in VMD. Ensure that size of sigma does not surprass this value.
    if len(sigma)>31: raise ValueError('Maximum of 32 bins allowed.')
    # Generate path to temporary xyz file
    xyzPath=os.path.join(os.path.dirname(__file__),
                         '_temp',
                         secrets.token_hex(15)+'.xyz')
    # Open xyz file
    with open(xyzPath,'w') as xyzFile:
        # Print total number of segments
        xyzFile.write(str(sigmaMatrix.shape[0])+'\n')
        # Iterate over number of bins
        for n in range(nBins):
            # Get number of unprinted segments
            nSeg=sigmaMatrix.shape[0]
            # Define local counter of printed segments
            nPrinted=0
            # Iterate over sigmaMatrix
            for seg in range(nSeg):
                # Check if current segment is within current iteration bin
                if sigma[n]-bins[2]/2<=sigmaMatrix[seg,5]<sigma[n]+bins[2]/2:
                    # Write to file
                    xyzFile.write('\nH'
                                  +str(n)
                                  +' '+str(sigmaMatrix[seg,0])
                                  +' '+str(sigmaMatrix[seg,1])
                                  +' '+str(sigmaMatrix[seg,2]))
                    # Update counter
                    nPrinted+=1
                # If outside bin, delete segments already printed and skip loop
                else:
                    sigmaMatrix=numpy.delete(sigmaMatrix,range(nPrinted),axis=0)
                    break
    # Convert xyzPath to VMD standard (windows to linux, \ to /)
    xyzPath=xyzPath.replace(os.sep,'/')
    # Generate path to temporary tcl script
    tclPath=os.path.join(os.path.dirname(__file__),
                         '_temp',
                         secrets.token_hex(15)+'.tcl')
    # Open tcl file
    with open(tclPath,'w') as tclFile:
        # Command to open xyz file
        tclFile.write('mol new {'+xyzPath+'} type {xyz}\n')
        # Iterate over bins
        for n in range(nBins):
            # Add rep
            tclFile.write('mol addrep 0\n')
            # Select Hn
            tclFile.write('mol modselect '+str(n)+' 0 name H'+str(n)+'\n')
            # Change color mode to ColorID with ID n
            tclFile.write('mol modcolor '+str(n)+' 0 ColorID '+str(n)+'\n')
            # Change material
            tclFile.write('mol modmaterial '+str(n)+' 0 Transparent\n')
            # Set style
            #tclFile.write('mol modstyle '+str(n)+' 0 QuickSurf 0.5 1 0.5 3\n')
            tclFile.write('mol modstyle '+str(n)+' 0 CPK 2 0 97 12\n')
            #tclFile.write('mol modstyle '+str(n)+' 0 VDW 1 132\n')
            # Calculate color scheme for Hn
            R=[1,0.8,0.6,0.4,0.2,0,0,0,0,0,0,0,0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1,1,1,0.9294,0.8588,0.7882,0.7176,0.647058824]
            G=[0,0,0,0,0,0,0.2,0.4,0.6,0.8,1,1,1,1,1,1,1,1,1,1,1,0.8,0.6,0.4,0.2,0,0.033,0.066,0.099,0.132,0.164705882]
            B=[1,1,1,1,1,1,1,1,1,1,1,0.8,0.6,0.4,0.2,0,0,0,0,0,0,0,0,0,0,0,0.033,0.066,0.099,0.132,0.164705882]
            R=R[n]
            G=G[n]
            B=B[n]

            # R=1/(1+numpy.exp(-(sigma[n]-0.01)*500))
            # B=1/(1+numpy.exp((sigma[n]+0.01)*500))
            # G=1-R-B
            # Change color scheme
            tclFile.write('color change rgb '+str(n)+' '+str(R)+' '+str(G)+' '+str(B)+'\n')
        # Delete last rep
        tclFile.write('mol delrep 31 0\n')
        # Set last color ID as white for background
        tclFile.write('color change rgb 32 1 1 1\n')
        tclFile.write('color Display Background orange3\n')
        # Load xyz
        if geometryXYZ is not None:
            tclFile.write('mol new {'+geometryXYZ+'} type {xyz}\n')
            tclFile.write('mol modstyle 0 1 CPK 1.000000 0.300000 102.000000 102.000000\n')
    # Run VMD using the tcl script as input
    subprocess.call('"'+vmdPath+'"'+' -e '+'"'+tclPath+'"')
    # Delete temorary files
    os.remove(xyzPath)
    os.remove(tclPath)
    # Output
    return None
    
def xyz2bmp(xyzPath,savePath,vmdPath):
    """
    xyz2bmp() generates a render (bmp image) of the XYZ file supplied using 
    VMD.

    Parameters
    ----------
    xyzPath : string
        Path to the XYZ file that is to be rendered.
    savePath : string
        Path for the bmp image to be saved. Must include the .bmp extension.
    vmdPath : string
        Path to the VMD executable.

    Returns
    -------
    None.

    """
    # Generate random temporary name for tcl script
    randomName=secrets.token_hex(15)
    # Get temporary folder location (lib/_temp) and generate temp tcl path
    tclPath=os.path.join(os.path.dirname(__file__),
                          '_temp',
                          randomName+'.tcl')
    # Convert xyzPath to VMD standard (windows to linux, \ to /)
    xyzPath=xyzPath.replace(os.sep,'/')
    # Convert savePath to VMD standard (windows to linux, \ to /)
    savePath=savePath.replace(os.sep,'/')
    # Write TCL script
    with open(tclPath,'w') as file:
        file.write('mol new {'
                   +xyzPath
                   +'} type {xyz} first 0 last -1 step 1 waitfor 1\n')
        file.write('mol modstyle 0 0 CPK 1.000000 0.300000 62.000000 62.000000'
                   +'\n')
        file.write('display resize 800 800\n')
        file.write('render snapshot '+'"'+savePath+'"'+'\n')
        file.write('exit')
    # Run VMD using the tcl script as input
    subprocess.call('"'+vmdPath+'"'+' -e '+'"'+tclPath+'"')
    # Delete tcl script
    os.remove(tclPath)
    # Output
    return None

def openVMD(xyzPath,vmdPath):
    
    
    # Generate random temporary name for tcl script
    randomName=secrets.token_hex(15)
    # Get temporary folder location (lib/_temp) and generate temp tcl path
    tclPath=os.path.join(os.path.dirname(__file__),
                          '_temp',
                          randomName+'.tcl')
    # Convert xyzPath to VMD standard (windows to linux, \ to /)
    xyzPath=xyzPath.replace(os.sep,'/')
    # Write TCL script
    with open(tclPath,'w') as file:
        file.write('mol new {'+xyzPath+'} type {xyz} first 0 last -1 step 1 waitfor 1\n')
        file.write('mol modstyle 0 0 CPK 1.000000 0.300000 62.000000 62.000000\n')
        file.write('display resize 800 800\n')
    # Run VMD using the tcl script as input
    subprocess.call('"'+vmdPath+'"'+' -e '+'"'+tclPath+'"') # Might not work outside of Windows; change in the future...
    # Delete tcl script
    os.remove(tclPath)
    # Output
    return None

def viewFinalGeometries(databaseFolder,vmdPath):
    
    
    # List folders in databaseFolder
    folderList=[]
    for obj in os.listdir(databaseFolder):
        if os.path.isdir(os.path.join(databaseFolder,obj)): folderList.append(os.path.join(databaseFolder,obj))
    # Generate path to temporary xyz file
    trajPath=os.path.join(os.path.dirname(__file__),
                          '_temp',
                          secrets.token_hex(15)+'.xyz')
    # Initialize list of viewed folders
    viewed=[]
    # Open xyz file
    with open(trajPath,'w') as trajFile:
        # Iterate over folderList
        for folder in tqdm(folderList,'Generating trajectory: '):
            # Check if folder has a finalGeometry file
            if os.path.isfile(os.path.join(folder,'finalGeometry.xyz')):
                viewed.append(os.path.basename(os.path.normpath(folder)))
                # Open xyz file
                with open(os.path.join(folder,'finalGeometry.xyz'),'r') as xyzFile:
                    # Get total number of atoms
                    nAtom=int(xyzFile.readline())
                    # Skip line
                    xyzFile.readline()
                    # Print total number of atoms
                    trajFile.write('2000\n\n')
                    # Iterate over xyzFile
                    for line in xyzFile:
                        trajFile.write(line)
                    # Padding to 2000
                    for n in range(2000-nAtom):
                        trajFile.write('J 0 0 0\n')
                    
    # Convert trajPath to VMD standard (windows to linux, \ to /)
    trajPath=trajPath.replace(os.sep,'/')
    # Open VMD
    openVMD(trajPath,vmdPath)
    # Remove temporary file
    os.remove(trajPath)
    # Output
    return viewed
