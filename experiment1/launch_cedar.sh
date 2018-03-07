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

 for s in 10 50
 do
    sbatch experiment1/launch_conll_cedar.sh 1 $s
    sbatch experiment1/launch_pos_cedar.sh 1 $s
    sbatch experiment1/launch_ner_cedar.sh 1 $s
    sbatch experiment1/launch_ocr_cedar.sh 1 $s
 done

