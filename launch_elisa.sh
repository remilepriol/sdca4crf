#!/usr/bin/env bash
#SBATCH --time=0:10:00
#SBATCH --job-name=sdca
#SBATCH --mem=1GB
#SBATCH --output=sdca_results.out
#SBATCH --qos=high
#SBATCH --gres=gpu:0


for i in 0.2 0.5 0.8 1.0
do
   sbatch launch_conll_elisa.sh $i
   sbatch launch_pos_elisa.sh $i
   sbatch launch_ner_elisa.sh $i
   sbatch launch_ocr_elisa.sh $i
done

