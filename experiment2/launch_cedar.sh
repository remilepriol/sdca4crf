#!/usr/bin/env bash
if [ $USER == "apiche" ]
 then
  echo $USER
  #SBATCH --account=rpp-bengioy
fi
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --mem=10GB
#SBATCH --job-name=sdca
#SBATCH --output=sdca_results.out
#SBATCH --qos=high
source activate py362

 for s in 1
 do
    sbatch experiment2/launch_conll_cedar.sh
    sbatch experiment2/launch_pos_cedar.sh
    sbatch experiment2/launch_ner_cedar.sh
    sbatch experiment2/launch_ocr_cedar.sh
 done

