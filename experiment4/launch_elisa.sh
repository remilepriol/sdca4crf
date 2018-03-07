#!/usr/bin/env bash
#SBATCH --time=0:10:00
#SBATCH --job-name=sdca
#SBATCH --mem=1GB
#SBATCH --output=sdca_results.out
#SBATCH --qos=high
#SBATCH --gres=gpu:0


for s in importance gap gap+ max
do
 for i in 0.0 0.2 0.5 0.8 1.0
 do
       sbatch experiment4/launch_conll_elisa.sh $i $s
       sbatch experiment4/launch_pos_elisa.sh $i $s
       sbatch experiment4/launch_ner_elisa.sh $i $s
       sbatch experiment4/launch_ocr_elisa.sh $i $s
 done
done

