#!/bin/bash
#$ -q hpc 
#$ -pe smp 30
#$ -N AcetylFentanyl

## ----------------------------------------------------------
## Set environment - MODIFY MODULE NAMES BEFORE RUNNING
export OMP_NUM_THREADS=${NSLOTS}
module purge
conda deactivate
conda activate spg-env
module load nwchem/7.2  

# Print the hostname of the machine the job is running on
echo "Job is running on machine: $(hostname)"

## ----------------------------------------------------------
# Set up molecule info
mol_idx=01
mol_name="Acetyl fentanyl"
# identifier_type="CAS-Number"
# identifier="3258-84-2"
# identifier_type="InChIKey"
# identifier="FYIUUQUPOKIKNI-UHFFFAOYSA-N"
identifier_type="SMILES"
identifier="CC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3"
initXYZ=None
charge=0

job_name="${mol_idx}-${identifier_type}"

# Print the molecule info
echo "mol_idx = ${mol_idx}"
echo "mol_name = ${mol_name}"
echo "identifier_type = ${identifier_type}"
echo "xyz file path = ${initXYZ}"
echo "identifier = ${identifier}"
echo "charge = ${charge}"


## ----------------------------------------------------------
## Run current task

# Get current local directory
curr=$(pwd)
# Create temp folder in node (tmp/XXX)
MY_TEMP=$(mktemp -d)
# Copy files to temp folder
cp -r Python "$MY_TEMP"
# Go into tmp/XXX/Python
cd "$MY_TEMP"
cd Python

# Run the generation script
python RunRepeats.py --idtype ${identifier_type} --id ${identifier} --charge ${charge} --name ${job_name} --nslots $NSLOTS

# Go back into tmp/XXX
cd ..
# Delete the unneded folders (everything except the job folder)
rm -rf Python
# Copy the job folder back to the current directory
cp -r * $curr
# Remove temp folder (tmp/XXX)
/bin/rm -r $MY_TEMP
# Return to current directory
cd $curr


