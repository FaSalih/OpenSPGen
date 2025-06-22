#!/bin/bash

conda activate gpflow-env

# train GP model for individual folds
for k in {0..9}; do (
    echo "Running fold $k"
    python train-gp-model.py $k
)

# combine/average GP model results from all folds
python train-gp-model.py all
