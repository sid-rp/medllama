#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=llama_galore_tuning
#SBATCH --account=pr_96_general
#SBATCH --time=15:00:00
#SBATCH --mem=90GB
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sp7835@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
source ~/.bashrc
mamba activate medllama

cd /scratch/sp7835/medAlpaca/
python medalpaca/train_new.py --finetuning_method galore

