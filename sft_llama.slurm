#!/bin/bash
#SBATCH --job-name=llama_sft_tuning
#SBATCH --account=ece_gy_9193_001-2024fa
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

singularity exec --bind /scratch --nv --overlay /scratch/sp7835/overlay-50G-10M.ext3:ro /scratch/sp7835/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate medalpaca
    cd /scratch/sp7835/medAlpaca
    python medalpaca/train.py \
        --model meta-llama/Llama-3.2-3B \
        --prompt_template medalpaca/prompt_templates/medalpaca.json \
        --data_path data/merged_medical_meadow.json \
        --val_set_size 0.1 \
        --output_dir 'output' \
        --train_in_8bit False \
        --model_max_length 256 \
        --use_lora False \
        --use_wandb False \
        --bf16 False \
        --tf32 True \
        --fp16 False \
        --global_batch_size 128 \
        --per_device_batch_size 8
"

