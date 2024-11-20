import json
from sklearn.metrics import classification_report

# Define file paths
steps = ["step1", "step2", "step3"]
model_path_template = "/scratch/sp7835/medAlpaca/usmle/{step}_Llama-3.2-3B.json"
gt_path_template = "/scratch/sp7835/medAlpaca/usmle/{step}_solutions.json"

# Loop through each step and evaluate
for step in steps:
    # Load ground truth answers
    gt_path = gt_path_template.format(step=step)
    with open(gt_path, 'r') as f:
        gt_answers = json.load(f)  # Format: {"1": "B", "2": "E", ...}

    # Load model output
    model_path = model_path_template.format(step=step)
    with open(model_path, 'r') as f:
        model_output = json.load(f)  # Format: [{"no": 1, "answer": "B"}, ...]

    # Extract model's answers in a comparable format
    model_answers = {str(item['no']): item['answer'].split(':')[0].strip() for item in model_output}

    # Evaluate accuracy
    correct = sum(1 for q, ans in model_answers.items() if gt_answers.get(q) == ans)
    total = len(gt_answers)
    accuracy = correct / total * 100

    print(f"Accuracy for {step}: {accuracy:.2f}%")

    # Create lists of the ground truth and predicted answers for classification report
    y_true = [gt_answers[str(q)] for q in model_answers.keys()]
    y_pred = [model_answers[str(q)] for q in model_answers.keys()]

    # Generate classification report
    report = classification_report(y_true, y_pred, labels=list("ABCDE"), zero_division=0)
    print(f"Classification Report for {step}:\n{report}\n")
