#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_conll
#SBATCH --mem=20GB
#SBATCH --output=conll_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362

python main.py --dataset conll --non-uniformity $1