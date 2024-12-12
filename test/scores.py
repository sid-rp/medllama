from evaluate import load
import json
import numpy as np

# Load the datasets module
references = []
predictions = []
with open('./generated_predictions.jsonl', 'r') as file:  
    for line in file:
        prediction = json.loads(line)
        references.append(prediction['label'])
        predictions.append(prediction['predict'])

def calculate_bertscore(references, predictions):
    """
    Calculate BERT Score for predictions against references
    
    Parameters:
    references (list): List of reference texts
    predictions (list): List of predicted texts
    
    Returns:
    dict: Dictionary containing the BERT Score metrics
    """
    # Load BERT Score metric
    bertscore = load("bertscore")
    
    # Calculate BERT Score
    bert_results = bertscore.compute(predictions=predictions, 
                                   references=references, 
                                   lang="en",
                                   model_type="microsoft/deberta-xlarge-mnli")  # Using a robust model
    
    # Calculate average BERT Score components
    avg_precision = np.mean(bert_results['precision'])
    avg_recall = np.mean(bert_results['recall'])
    avg_f1 = np.mean(bert_results['f1'])
    
    # Compile results
    results = {
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1': float(avg_f1)
    }
    
    return results

# Calculate and display the metrics
results = calculate_bertscore(references, predictions)

# Print results in a formatted way
print("\nBERT Score Results:")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1: {results['f1']:.4f}")