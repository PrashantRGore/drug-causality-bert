# src/inference.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import PyPDF2
import numpy as np
import re
from nltk.tokenize import sent_tokenize
import nltk

class CausalityClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text, threshold=0.5):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        pred = 1 if probs[1] > threshold else 0
        return {
            'prediction': 'related' if pred == 1 else 'not related',
            'confidence': float(probs[pred]),
            'probabilities': {
                'not_related': float(probs[0]),
                'related': float(probs[1])
            }
        }

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)]', '', text)
    text = ' '.join([w for w in text.split() if len(w) > 1])
    return text.strip()

def classify_causality(pdf_text, threshold=0.5, max_length=96, batch_size=32):
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    cleaned_text = preprocess_text(pdf_text)
    sentences = sent_tokenize(cleaned_text)
    model_path = "models/production_model_final"
    classifier = CausalityClassifier(model_path)
    classifications, sentence_details = [], []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sentences))
            batch_sentences = sentences[start_idx:end_idx]
            inputs = classifier.tokenizer(
                batch_sentences,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            outputs = classifier.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            for i, sent in enumerate(batch_sentences):
                prob_related = probs[i][1].item()
                pred = 1 if prob_related > threshold else 0
                classifications.append(pred)
                sentence_details.append({
                    'sentence': sent[:100] + ('...' if len(sent) > 100 else ''),
                    'probability_related': prob_related,
                    'prediction': 'related' if pred == 1 else 'not related',
                    'confidence': max(prob_related, 1 - prob_related)
                })
    num_related = sum(classifications)
    num_not_related = len(classifications) - num_related
    final_classification = "related" if num_related > 0 else "not related"
    confidence_score = num_related / len(classifications) if classifications else 0
    sentence_details.sort(key=lambda x: x['probability_related'], reverse=True)
    results = {
        'final_classification': final_classification,
        'confidence_score': confidence_score,
        'related_sentences': num_related,
        'not_related_sentences': num_not_related,
        'total_sentences': len(sentences),
        'top_related_sentences': sentence_details[:5],
        'threshold_used': threshold
    }
    return results

def process_pdf_file(pdf_path, threshold=0.5):
    pdf_text = extract_text_from_pdf(pdf_path)
    results = classify_causality(pdf_text, threshold=threshold)
    results["pdf_file"] = Path(pdf_path).name
    return results

def process_multiple_pdfs(pdf_paths, threshold=0.5):
    all_results = []
    for pdf_path in pdf_paths:
        try:
            res = process_pdf_file(pdf_path, threshold)
            all_results.append(res)
        except Exception as e:
            all_results.append({"pdf_file": Path(pdf_path).name, "error": str(e), "final_classification": "error"})
    return all_results
