import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import PyPDF2
import json
from datetime import datetime
from typing import Union, List, Dict
import re

# NLTK with robust error handling
import nltk
import ssl

# SSL fix for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Enhanced NLTK data download with retry
def download_nltk_data_robust():
    """Download NLTK data with multiple attempts and fallbacks"""
    import os
    
    # Set NLTK data path explicitly
    nltk_data_dir = '/home/appuser/nltk_data'
    if not os.path.exists(nltk_data_dir):
        try:
            os.makedirs(nltk_data_dir, exist_ok=True)
        except:
            pass
    
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    packages = ['punkt', 'punkt_tab']
    for package in packages:
        for attempt in range(3):  # Try 3 times
            try:
                nltk.data.find(f'tokenizers/{package}')
                print(f"✓ {package} already available")
                break
            except LookupError:
                try:
                    print(f"Downloading {package} (attempt {attempt + 1})...")
                    nltk.download(package, download_dir=nltk_data_dir, quiet=False)
                    print(f"✓ {package} downloaded successfully")
                    break
                except Exception as e:
                    print(f"Warning: Could not download {package}: {e}")
                    if attempt == 2:
                        print(f"Failed to download {package} after 3 attempts")

# Download on import
download_nltk_data_robust()

# Fallback sentence tokenizer using regex
def simple_sentence_tokenize(text):
    """Simple regex-based sentence tokenizer as fallback"""
    # Split on common sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Safe sentence tokenization with fallback
def safe_sent_tokenize(text):
    """Tokenize with NLTK, fallback to regex if NLTK fails"""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception as e:
        print(f"NLTK tokenization failed ({e}), using fallback...")
        return simple_sentence_tokenize(text)

class CausalityClassifier:
    def __init__(self, model_path='./models/production_model_final', threshold=0.5):
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
        result = {
            'prediction': 'related' if pred == 1 else 'not related',
            'confidence': float(probs[pred]),
            'label': int(pred)
        }
        if return_probs:
            result['probabilities'] = {
                'not_related': float(probs[0]),
                'related': float(probs[1])
            }
        return result

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def classify_causality(pdf_text, model_path='./models/production_model_final', threshold=0.5, verbose=False):
    classifier = CausalityClassifier(model_path, threshold)
    
    # Use safe tokenization with fallback
    sentences = safe_sent_tokenize(pdf_text)
    
    if verbose:
        print(f"Tokenized {len(sentences)} sentences")
    
    related_count = 0
    sentence_details = []
    
    for sent in sentences:
        if not sent.strip():
            continue
            
        result = classifier.predict(sent, return_probs=True)
        if result['label'] == 1:
            related_count += 1
            sentence_details.append({
                'sentence': sent[:100],
                'probability_related': result['probabilities']['related'],
                'confidence': result['confidence']
            })
    
    sentence_details.sort(key=lambda x: x['probability_related'], reverse=True)
    
    return {
        'final_classification': 'related' if related_count > 0 else 'not related',
        'confidence_score': related_count / len(sentences) if sentences else 0,
        'related_sentences': related_count,
        'total_sentences': len(sentences),
        'top_related_sentences': sentence_details[:5],
        'threshold_used': threshold
    }

def process_pdf_file(pdf_path, model_path='./models/production_model_final', threshold=0.5, save_report=False, output_dir='./results'):
    pdf_text = extract_text_from_pdf(pdf_path)
    results = classify_causality(pdf_text, model_path, threshold)
    results['pdf_file'] = str(Path(pdf_path).name)
    if save_report:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{Path(pdf_path).stem}_report.json", 'w') as f:
            json.dump(results, f, indent=2)
    return results

def process_multiple_pdfs(pdf_paths, model_path='./models/production_model_final', threshold=0.5, save_reports=False, output_dir='./results'):
    all_results = []
    for pdf_path in pdf_paths:
        try:
            results = process_pdf_file(pdf_path, model_path, threshold, save_reports, output_dir)
            all_results.append(results)
        except Exception as e:
            all_results.append({
                'pdf_file': str(Path(pdf_path).name),
                'error': str(e),
                'final_classification': 'error'
            })
    return all_results
