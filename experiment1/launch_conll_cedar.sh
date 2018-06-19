#!/usr/bin/env bash
if [ $USER == "apiche" ]
 then
  echo $USER
  #SBATCH --account=rpp-bengioy
fi
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_conll
#SBATCH --mem=20GB
#SBATCH --output=conll_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362

python main.py --line-search scipy --dataset conll --non-uniformity $1 --sampler-period $2

