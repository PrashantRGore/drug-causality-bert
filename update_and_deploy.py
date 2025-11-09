# update_and_deploy.py
import subprocess
from pathlib import Path

# Complete inference.py content
inference_code = '''import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import PyPDF2
from nltk.tokenize import sent_tokenize
import nltk
import json
from datetime import datetime
from typing import Union, List, Dict

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class CausalityClassifier:
    def __init__(self, model_path='PrashantRGore/drug-causality-bert-model', threshold=0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
    
    def predict(self, text, return_probs=False):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            pred = 1 if probs[1] > self.threshold else 0
        result = {'prediction': 'related' if pred == 1 else 'not related', 'confidence': float(probs[pred]), 'label': int(pred)}
        if return_probs:
            result['probabilities'] = {'not_related': float(probs[0]), 'related': float(probs[1])}
        return result

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def classify_causality(pdf_text, model_path='PrashantRGore/drug-causality-bert-model', threshold=0.5, verbose=False):
    classifier = CausalityClassifier(model_path, threshold)
    sentences = sent_tokenize(pdf_text)
    related_count = 0
    sentence_details = []
    for sent in sentences:
        result = classifier.predict(sent, return_probs=True)
        if result['label'] == 1:
            related_count += 1
            sentence_details.append({'sentence': sent[:100], 'probability_related': result['probabilities']['related'], 'confidence': result['confidence']})
    sentence_details.sort(key=lambda x: x['probability_related'], reverse=True)
    return {'final_classification': 'related' if related_count > 0 else 'not related', 'confidence_score': related_count / len(sentences) if sentences else 0, 'related_sentences': related_count, 'total_sentences': len(sentences), 'top_related_sentences': sentence_details[:5], 'threshold_used': threshold}

def process_pdf_file(pdf_path, model_path='PrashantRGore/drug-causality-bert-model', threshold=0.5, save_report=False, output_dir='./results'):
    pdf_text = extract_text_from_pdf(pdf_path)
    results = classify_causality(pdf_text, model_path, threshold)
    results['pdf_file'] = str(Path(pdf_path).name)
    if save_report:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{Path(pdf_path).stem}_report.json", 'w') as f:
            json.dump(results, f, indent=2)
    return results

def process_multiple_pdfs(pdf_paths, model_path='PrashantRGore/drug-causality-bert-model', threshold=0.5, save_reports=False, output_dir='./results'):
    all_results = []
    for pdf_path in pdf_paths:
        try:
            results = process_pdf_file(pdf_path, model_path, threshold, save_reports, output_dir)
            all_results.append(results)
        except Exception as e:
            all_results.append({'pdf_file': str(Path(pdf_path).name), 'error': str(e), 'final_classification': 'error'})
    return all_results
'''

# Write the file
print("Writing src/inference.py...")
Path("src/inference.py").write_text(inference_code, encoding='utf-8')
print("✓ File written")

# Git commands
commands = [
    ["git", "add", "src/inference.py"],
    ["git", "commit", "-m", "Add complete inference module with PDF processing"],
    ["git", "push"]
]

print("\nExecuting git commands...")
for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Success")
    else:
        print(f"✗ Error: {result.stderr}")

print("\n✓ Done! Streamlit will auto-redeploy in 2-3 minutes.")
