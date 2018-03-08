#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --job-name=sdca_ocr
#SBATCH --mem=10GB
#SBATCH --output=ocr_results.out
#SBATCH --qos=low
#SBATCH --gres=gpu:0

source activate py362
for s in ocr conll
do
 for i in 0.0 0.8 1.0
 do
       sbatch experiment5/launch_dataset.sh $s $i
 done
done

