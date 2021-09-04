#!/bin/bash
#SBATCH --job-name clear-rain-classfication
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv trial.sif python train.py
date