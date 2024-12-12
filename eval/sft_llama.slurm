#!/bin/bash
#SBATCH --job-name=llama_sft_tuning
#SBATCH --account=ece_gy_9193_001-2024fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sp7835@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/ym3144/overlay-25G-10M.ext3:ro /scratch/ym3144/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate medalpaca
    cd /scratch/ym3144/medAlpaca
    python eval_usmle.py \
        --model_name 'meta-llama/Llama-3.2-3B' \
        --prompt_template '../medalpaca/prompt_templates/med_llama3.json' \
        --base_model 'decapoda-research/llama-13b-hf' \
        --peft True \
        --load_in_8bit True \
        --path_to_exams '/scratch/sp7835/medAlpaca/usmle'
"
