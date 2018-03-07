#!/usr/bin/env bash
if [ $USER == "apiche" ]
 then
  echo $USER
  #SBATCH --account=rpp-bengioy
fi
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_pos
#SBATCH --mem=40GB
#SBATCH --output=pos_results.out
#SBATCH --gres=gpu:0

source activate py362

python main.py --line-search scipy --dataset pos --non-uniformity $1 --sampler-period $2

