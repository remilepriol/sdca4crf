#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=sdca
#SBATCH --mem=10GB
#SBATCH --output=sdca_results.out
#SBATCH --qos=high
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL --mail-user=alex_piche_l@hotmail.com

source activate py362

python sdca4crf/main.py --dataset $1
