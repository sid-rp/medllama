# Medical QA Fine-Tuning Project

A comprehensive study on fine-tuning LLaMA 3.2 3B for medical question-answering tasks, exploring various techniques including LoRA, QLoRA, GaLore, and KnowTuning.

## Overview

This project focuses on optimizing the Meta LLaMA 3.2 3B model for medical question-answering tasks using a combination of supervised and parameter-efficient fine-tuning techniques. Our goal is to enhance the model's ability to provide accurate and contextually relevant responses to medical queries.

## Features

- Implementation of multiple fine-tuning techniques:
  - Supervised Fine-Tuning (SFT)
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - GaLore (Gradient-based Low-Rank Adaptation)
  - KnowTuning (Knowledge-focused tuning)

## Dataset Integration

The project utilizes diverse medical datasets:
- CORD-19: COVID-19 research literature
- MediQA and MedQA: Clinical scenario questions
- MMLU Medical Subset: Comprehensive medical knowledge
- WikiDoc Medical Flashcards: Quick medical fact retrieval
- MedQuAD: Structured medical questions from trusted sources

## Technical Configuration

- Base Model: Meta LLaMA 3.2 3B
- Max Token Length: 256
- Batch Size: 4
- Learning Rate: 2e-5
- Training Environment: NVIDIA A100 GPU
- Training Duration: ~13-14 hours per epoch
- Training Steps: 33,590 (with gradient accumulation)

## Performance Results

### USMLE Performance
| Model    | Step 1 | Step 2 | Step 3 |
|----------|--------|--------|--------|
| Baseline | 0.39   | 0.29   | 0.38   |
| SFT      | 0.39   | 0.32   | 0.42   |
| LoRA     | 0.39   | 0.33   | 0.36   |
| QLoRA    | 0.38   | 0.38   | 0.38   |
| GaLore   | 0.42   | 0.34   | 0.45   |

### MedQuAD Results
| Model      | Precision | Recall | F1 Score |
|------------|-----------|--------|-----------|
| Baseline   | 0.56      | 0.57   | 0.56      |
| SFT        | 0.69      | 0.69   | 0.69      |
| LoRA       | 0.70      | 0.68   | 0.69      |
| QLoRA      | 0.72      | 0.68   | 0.70      |
| GaLore     | 0.71      | 0.69   | 0.70      |
| KnowTuning | 0.69      | 0.66   | 0.67      |

## Installation & Usage

### Getting Started

creating a new conda environment

```
conda create -n medllama "python>=3.9"
```

```
pip install -r requirements.txt
```

### Training steps

```
python3 src/train.py 
 --prompt_template <val>
 --model_max_length <val>
 --train_on_inputs <val>
 --data_path <val>
 --val_set_size <val>
 --batch_size <val>
 --epochs <val>
 --learning_rate <val>
 --finetuning_method <val>
```


## Authors

- Madhuri Mahalingam (New York University)
- Yashaswi Makula (New York University)
- Sidhartha Reddy Potu (New York University)

## Citation

If you use this work, please cite:
```bibtex
@article{medqa2024,
  title={A Study of Fine Tuning approaches in the Domain of Medicine Q&A},
  author={Mahalingam, Madhuri and Makula, Yashaswi and Potu, Sidhartha Reddy},
  institution={New York University},
  year={2024}
}
```
