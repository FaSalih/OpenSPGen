# NWChem Installation Instruction
The following is a set of NWChem installation istructions that were used in the **specific case of the author's HPC machine** in hope that they may help someone trying to install the same version of NWChem on a linux system.

## Machine Information:
1. Name and version
    - `NAME="Red Hat Enterprise Linux"`
    - `VERSION="9.6 (Plow)"`
2. Kernel
    - `5.14.0-570.21.1.el9_6.x86_64`
3. CPU Information
    - `model name      : AMD EPYC 7543 32-Core Processor`

## Installation Instructions
After cloning the desired NWChem version repo, we execute the following in our front-end mahcine:
```
module load ompi/3.0.0/intel/18.0          # module may or may not exist
export LARGE_FILES=TRUE
export TCGRSH="/usr/kerberos/bin/rsh -F"   # path may vary
export NWCHEM_TOP=`pwd`
export NWCHEM_TARGET=LINUX64               # machine may vary
export MPI_LOC=/afs/crc.nd.edu/x86_64_linux/o/openmpi/3.0.0/intel/18.0   # path may vary
export USE_MPI=y
export USE_MPIF=y
export USE_MPIF4=y
export ARMCI_NETWORK=MPI-PR
export FC=mpif90
export CC=mpicc
export CXX=mpicxx
export F77=mpif77
export USE_INTERNALBLAS="y"
export USE_NOFSCHECK=TRUE
export MRCC_METHODS=TRUE
```
We would then go to the NWChem `src` directory, configure NWChem, compile, and link it:
```
cd $NWCHEM_TOP/src
make nwchem_config NWCHEM_MODULES=all
make -j4 >& make.log &
```
