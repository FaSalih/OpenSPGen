"""
Visualizes a sigma surface for a terminal-proveded cosmo.xyz file using VMD
"""

import sys
import subprocess
sys.path.append('Python')
from lib import VMD_Wrapper as vmd

# check for correct number of arguments
if len(sys.argv) != 2:
    print("Usage: python3 visualize_sigma_surface.py <external molecule folder>")
    sys.exit(1)

# get the external folder name
folder = sys.argv[1]

# build the job name
job_name = f"{folder}_0"

# get the file names
surfaceCSV = f"{folder}/{job_name}/sigmaSurface.csv"
geometryXYZ = f"{folder}/{job_name}/finalGeometry.xyz"
vmdPath = "" # module will be loaded

# load vmd module in CRC
subprocess.call("module load vmd",shell=True)

# run the visualization
vmd.viewSigmaSurface(surfaceCSV,geometryXYZ,vmdPath,avgRadius=0.5,bins=[-0.100,0.100,0.010])
