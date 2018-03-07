#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_ner
#SBATCH --mem=20GB
#SBATCH --output=ner_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362

python main.py --dataset ner --non-uniformity $1 --sampling-scheme $2
