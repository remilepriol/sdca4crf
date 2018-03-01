#!/usr/bin/env bash
#SBATCH --account=rpp-bengioy
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --job-name=sdca
#SBATCH --output=sdca_results.out
#SBATCH --qos=high

source activate py362

python sdca4crf/main.py --dataset dataset
