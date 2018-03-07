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

 for s in True
 do
    for i in 5e-1 1e-1 1e-2
    do
       sbatch experiment3/launch_conll_cedar.sh $i
       sbatch experiment3/launch_pos_cedar.sh $i
       sbatch experiment3/launch_ner_cedar.sh $i
       sbatch experiment3/launch_ocr_cedar.sh $i
    done
 done
