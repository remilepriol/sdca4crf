#!/usr/bin/env bash
#SBATCH --account=def-bengioy
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --mem=20GB
#SBATCH --job-name=sdca
#SBATCH --output=sdca_results.out
#SBATCH --qos=high

source activate py362

python sdca4crf/main.py --dataset $1
