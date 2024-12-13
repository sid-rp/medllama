#!/bin/bash
#SBATCH --job-name=llama_sft_tuning
#SBATCH --account=ds_ga_1011-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sp7835@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/sp7835/overlay-25G-10M.ext3:ro /scratch/sp7835/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    mamba activate medllama
    cd /scratch/ym3144/medAlpaca
    python eval_usmle.py \
        --model_name 'meta-llama/Llama-3.2-3B' \
        --prompt_template 'src/prompt_templates/medalpaca_new.json' \
        --path_to_exams '../src/usmle'
"
