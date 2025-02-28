#!/bin/bash

## ----------------------------------------------------------
# This file runs aset of example jobs for Methane with different 
# input formats and options

## ----------------------------------------------------------
## Set environment - MODIFY MODULE NAMES BEFORE RUNNING
export OMP_NUM_THREADS=${NSLOTS}
module purge
conda deactivate
conda activate spg-env
module load nwchem/7.2

## ----------------------------------------------------------
## Set up all test parameters
mol_name="Methane"
charge=0
identifier_types=("CAS-Number" "InChIKey" "SMILES" "XYZ") 
identifiers=("74-82-8" "VNWKTOKETHGBQD-UHFFFAOYSA-N" "C" None)
initXYZs=(None None None "manuscript-databases/VT-2005_XYZs/VT2005-1.xyz")

## ----------------------------------------------------------
## Loop over tests
for i in {0..3}
do

    # Set up molecule info
    identifier_type=${identifier_types[$i]}
    identifier=${identifiers[$i]}
    initXYZ=${initXYZs[$i]}
    job_name="${mol_name}-${identifier_type}"
    
    # Print the test info
    echo -e "----------------------------------------------------------"
    echo -e "\nRunning test ${i} using ${identifier_type} input"
    echo -e "\tjob name = ${job_name}"
    echo -e "\tidentifier_type = ${identifier_type}"
    echo -e "\tidentifier = ${identifier}"
    echo -e "\txyz file path = ${initXYZ}"
    
    ## ----------------------------------------------------------
    ## Run current task

    # Get current local directory
    curr=$(pwd)
    # Make a directory for all test results
    mkdir ${curr}/tests/
    # Create temp folder in node (tmp/XXX)
    MY_TEMP=$(mktemp -d)
    # Copy files to temp folder
    cp -r manuscript-databases "$MY_TEMP"
    cp -r Python "$MY_TEMP"
    # Go into tmp/XXX/Python
    cd "$MY_TEMP"
    cd Python

    # Run the generation script
    python RunRepeats.py --idtype ${identifier_type} --id ${identifier} --charge ${charge} --name ${job_name} --nslots $NSLOTS --initialxyz ${initXYZ}

    # Go back into tmp/XXX
    cd ..
    # Delete the unneded folders (everything except the job folder)
    rm -rf Python manuscript-databases
    # Copy the job folder back to the current directory
    cp -r * ${curr}/tests/
    # Remove temp folder (tmp/XXX)
    /bin/rm -r $MY_TEMP
    # Return to current directory
    cd ${curr}
done



