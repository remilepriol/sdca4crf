#!/usr/bin/env bash
if [ $USER == "apiche" ]
 then
  echo $USER
  #SBATCH --account=rpp-bengioy
fi
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_ner
#SBATCH --mem=20GB
#SBATCH --output=ner_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362

python main.py --dataset ner --non-uniformity 0.8 --use-previous-step-size True