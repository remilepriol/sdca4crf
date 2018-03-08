#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_ocr
#SBATCH --mem=10GB
#SBATCH --output=ocr_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

python main.py --dataset $1 --non-uniformity $2 --sampling-scheme gap
