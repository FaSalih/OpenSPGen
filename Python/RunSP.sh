#!/bin/bash
#$ -q hpc
#$ -pe smp 32

## ----------------------------------------------------------
## Set environment
export OMP_NUM_THREADS=${NSLOTS}
module purge
conda deactivate
conda activate cursed
export PATH="/afs/crc.nd.edu/user/f/fsalih/Desktop/nwchem-7.2.0-beta2/bin/LINUX64:$PATH"
export NWCHEM_NWPW_LIBRARY=/afs/crc.nd.edu/user/f/fsalih/Desktop/nwchem-7.2.0-beta2/src/nwpw/libraryps
export NWCHEM_BASIS_LIBRARY=/afs/crc.nd.edu/user/f/fsalih/Desktop/nwchem-7.2.0-beta2/src/basis/libraries
module load ompi/3.0.0/intel/18.0

## ----------------------------------------------------------
## Create arrays of molecule indices and identifiers

# Initialize arrays
mol_idxs=()
cas_nos=()
mol_names=()
mol_smiles=()

# Read molecule info from CSV
filePath="Python/molecules.csv"
counter=0
while read line
do
   # Extract data from line
   mol_idx=$(echo "$line" | cut -d',' -f1)
   cas_no=$(echo "$line" | cut -d',' -f2)
   mol_name=$(echo "$line" | cut -d',' -f3)
   smiles=$(echo "$line" | cut -d',' -f4)

   # Save to arrays
   mol_idxs+=("$mol_idx")
   cas_nos+=("$cas_no")
   mol_names+=("$mol_name")
   mol_smiles+=("$mol_smiles")

   # Increment counter
   counter=$((counter+1))
done < "$filePath"

## ----------------------------------------------------------
## Get Current Task Details

# Get index for current task
i="${SGE_TASK_ID-1}"
echo i = $i

# Set path to XYZ folder
XYZ-Folder="Python/molecule-XYZs"

# Get molecule info for current task
mol_idx="${mol_idxs[$i]}"
mol_name="${mol_names[$i]}"
cas_no="${cas_nos[$i]}"
initXYZ="${XYZ-Folder}/molecule-${mol_idx}.xyz"
smiles="${mol_smiles[$i]}"

echo "mol_idx = ${mol_idx}"
echo "mol_name = ${mol_name}"
echo "CAS No = ${cas_no}"
echo "xyz file path = ${initXYZ}"
echo "smiles = ${smiles}"

## ----------------------------------------------------------
## Run Current Task in Temp Folder

# Get current local directory
curr=$(pwd)

# Create temp folder in node (tmp/XXX)
MY_TEMP=$(mktemp -d)

# Copy files to temp folder
cp -r Python "$MY_TEMP"

# Go into tmp/XXX/Python
cd "$MY_TEMP"
cd Python

# Run the optimization
python Run_Repeats.py --identifier $smiles --identifier_type SMILES --molecule_id $mol_idx --cas_number $cas_no --nslots $NSLOTS

# Go back into tmp/XXX
cd ..

# Delete the Python folder
rm -rf Python
rm -rf ${XYZ-Folder}

# If size of current directory is greater than 2MB (due to a failed job), then only copy output.nw, input*, and xyz files; otherwise copy everything
if [ $(du -s | cut -f1) -gt 1500 ]; then
    cp output.nw input* *xyz $curr
else
    cp -r * $curr
fi

# Remove temp folder (tmp/XXX)
/bin/rm -r $MY_TEMP


