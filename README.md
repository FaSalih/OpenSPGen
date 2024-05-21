# SPG
Open source sigma profile generator

## Installation Instructions
1. Install the open source DFT package `NWChem`. The version used during the development of this package is `7.2.0` available for download [here](https://github.com/nwchemgit/nwchem/releases/tag/v7.2.0-release)*.
2. Add the path of the `nwchem` executable to your `PATH` variable (the `nwchem` executable path should be along the lines of: `User/Desktop/nwchem-7.2.0/bin/LINUX64`)
3. Download the current repository to your local machine.
4. Create a conda environment where you can install `rdkit` and its dependencies from the provided `yml` file using the following instructions:
   ```
   # Go to the directory where the repository was installed
   cd <SPG-installation-path>
   cd Python
   # Create a conda environment for all the dependencies
   conda env create -n SPG --file crc-cursed.yml
   ```
5. Run the installation tests (will run a sigma profile generation job for methane with different inputs - a SMILES, a CAS number and a pre-optimized xyz).

*Note: Because the DFT software used in this package is only available for Linux and macOS distributions, the complete tool can only be run and should be installed on those machines. 

