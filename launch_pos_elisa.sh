#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_pos
#SBATCH --mem=40GB
#SBATCH --output=pos_results.out
#SBATCH --qos=high
#SBATCH --gres=gpu:0

source activate py362

python main.py --dataset pos  --non-uniformity $1
