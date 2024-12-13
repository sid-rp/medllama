import sys
import os
sys.path.append("..")

import re
import json
import fire
import string
import torch
from tqdm.autonotebook import tqdm
from medalpaca.inferer import Inferer

# Generation parameters
sampling = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9
}

def format_question(d): 
    question = d["question"]
    options = d["options"]
    for k, v in options.items(): 
        question += f"\n{k}: {v}"
    return question

def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str
    
    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""

def starts_with_capital_letter(input_str):
    """
    The answers should start like this: 
        'A: '
        'A. '
        'A '
    """
    pattern = r'^[A-Z](:|\.|) .+'
    return bool(re.match(pattern, input_str))

def main(
    model_name: str,
    prompt_template: str = "../medalpaca/prompt_templates/medalpaca_new.json",
    base_model: str = "meta-llama/Llama-3.2-3B",
    finetuning_method: str = "sft",
    model_max_length: int = 256,
    path_to_exams: str = "eval/data/test/",
    ntries: int = 5,
    skip_if_exists: bool = True,
):
    # Initialize model with appropriate parameters based on finetuning method
    model = Inferer(
        model_name=model_name,
        prompt_template=prompt_template,
        base_model=base_model,
        model_max_length=model_max_length,
        finetuning_method=finetuning_method
    )
    
    for step_idx in [1, 2, 3]: 
        with open(os.path.join(path_to_exams, f"step{step_idx}.json")) as fp: 
            step = json.load(fp)   
        outname = os.path.join(path_to_exams, f"step{step_idx}_{model_name.split('/')[-1]}.json")
        
        if os.path.exists(outname): 
            with open(outname, "r") as fp:
                answers = json.load(fp)
        else: 
            answers = []
        
        pbar = tqdm(step)
        pbar.set_description_str(f"Evaluating USMLE Step {step_idx}")
        
        for i, question in enumerate(pbar):
            if skip_if_exists and (i+1) <= len(answers):
                continue
                
            for j in range(ntries):
                response = model(
                    instruction="Answer this multiple-choice question by selecting only the correct answer letter (e.g., 'A', 'B', etc.).",
                    input=format_question(question),
                    output="Answer:",
                    **sampling
                )
                response = strip_special_chars(response)
                if starts_with_capital_letter(response):
                    pbar.set_postfix_str("")
                    break
                else:
                    pbar.set_postfix_str(f"Output not satisfactory, retrying {j+1}/{ntries}")
                    
            question["answer"] = response
            answers.append(question)
            with open(outname, "w+") as fp:
                json.dump(answers, fp)

if __name__ == "__main__":
    fire.Fire(main)