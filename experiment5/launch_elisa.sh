#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_ocr
#SBATCH --mem=10GB
#SBATCH --output=ocr_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362

python main.py --dataset ocr --non-uniformity 1. --sampling-scheme gap
python main.py --dataset ocr --non-uniformity 0.8 --sampling-scheme gap
python main.py --dataset ocr --non-uniformity 0. --sampling-scheme uniform

python main.py --dataset conll --non-uniformity 1. --sampling-scheme gap
python main.py --dataset conll --non-uniformity 0.8 --sampling-scheme gap
python main.py --dataset conll --non-uniformity 0. --sampling-scheme uniform
