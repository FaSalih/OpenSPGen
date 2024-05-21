# SPG
Open source sigma profile generator

## Installation Instructions
1. Install the open source DFT package `NWChem`. The version used during the development of this package is `7.2.0` available for download [here](https://github.com/nwchemgit/nwchem/releases/tag/v7.2.0-release).
2. Add the path of the `nwchem` executable to your `PATH` variable (the `nwchem` executable path should be along the lines of: `User/Desktop/nwchem-7.2.0/bin/LINUX64`)
3. Create a conda environment where you can install `rdkit` and its dependencies from the provided `yml` file using the following instructions:
   ```
   conda env create -n ENVNAME --file crc-cursed.yml
   ```
4. Download the current repository to your local machine.
5. Run the installation tests (will run a sigma profile generation job for methane with different inputs - a SMILES, a CAS number and a pre-optimized xyz).


