#!/usr/bin/env bash
#SBATCH --time=0:10:00
#SBATCH --job-name=sdca
#SBATCH --mem=1GB
#SBATCH --output=sdca_results.out
#SBATCH --qos=high
#SBATCH --gres=gpu:0


for s in uniform importance gap gap+ max
do
   sbatch experiment4/launch_conll_elisa.sh 0.8 $s
   sbatch experiment4/launch_pos_elisa.sh 0.8 $s
   sbatch experiment4/launch_ner_elisa.sh 0.8 $s
   sbatch experiment4/launch_ocr_elisa.sh 0.8 $s
done

