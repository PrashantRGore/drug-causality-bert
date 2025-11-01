"""
Drug-Adverse Event Causality Classification
Complete inference module with PDF processing capabilities
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import PyPDF2
from nltk.tokenize import sent_tokenize
import nltk
import json
from datetime import datetime
from typing import Union, List, Dict

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class CausalityClassifier:
    """
    Drug-Adverse Event Causality Classifier using BioBERT
    
    Args:
        model_path: Path to trained model directory
        threshold: Classification threshold (default: 0.5)
    """
    
    def __init__(self, model_path='./models/production_model_final', threshold=0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
    
    def predict(self, text, return_probs=False):
        """
        Predict causality for single text
        
        Args:
            text: Input text
            return_probs: Return probability distribution
            
        Returns:
            dict with prediction, confidence, and optionally probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=96
        )
        
        # Predict
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
    
    def predict_batch(self, texts):
        """Predict causality for multiple texts"""
        return [self.predict(text, return_probs=True) for text in texts]


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            return text
            
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error extracting PDF: {e}")


def classify_causality(
    pdf_text: str,
    model_path: str = './models/production_model_final',
    threshold: float = 0.5,
    max_length: int = 96,
    verbose: bool = False
) -> Dict:
    """
    Classify causality relationship in text
    
    Args:
        pdf_text: Extracted text to classify
        model_path: Path to trained model
        threshold: Classification threshold (0-1)
        max_length: Maximum sequence length
        verbose: Print progress
        
    Returns:
        Dictionary with classification results
    """
    
    start_time = datetime.now()
    
    if verbose:
        print(f"\nClassifying causality...")
        print(f"Text length: {len(pdf_text)} characters")
    
    # Initialize classifier
    classifier = CausalityClassifier(model_path, threshold)
    
    # Tokenize into sentences
    sentences = sent_tokenize(pdf_text)
    
    if verbose:
        print(f"Total sentences: {len(sentences)}")
    
    # Classify each sentence
    related_count = 0
    sentence_details = []
    
    for sent in sentences:
        result = classifier.predict(sent, return_probs=True)
        
        if result['label'] == 1:
            related_count += 1
            sentence_details.append({
                'sentence': sent[:100] + ('...' if len(sent) > 100 else ''),
                'probability_related': result['probabilities']['related'],
                'confidence': result['confidence']
            })
    
    # Sort by probability
    sentence_details.sort(key=lambda x: x['probability_related'], reverse=True)
    
    # Final classification
    final_classification = 'related' if related_count > 0 else 'not related'
    confidence_score = related_count / len(sentences) if sentences else 0
    
    # Processing time
    duration = (datetime.now() - start_time).total_seconds()
    
    results = {
        'final_classification': final_classification,
        'confidence_score': confidence_score,
        'related_sentences': related_count,
        'not_related_sentences': len(sentences) - related_count,
        'total_sentences': len(sentences),
        'top_related_sentences': sentence_details[:5],  # Top 5
        'threshold_used': threshold,
        'processing_time_seconds': duration,
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Classification: {final_classification}")
        print(f"  Confidence: {confidence_score:.2%}")
        print(f"  Related sentences: {related_count}/{len(sentences)}")
        print(f"  Processing time: {duration:.2f}s")
    
    return results


def process_pdf_file(
    pdf_path: str,
    model_path: str = './models/production_model_final',
    threshold: float = 0.5,
    save_report: bool = False,
    output_dir: str = './results'
) -> Dict:
    """
    Complete pipeline: Extract PDF → Classify → Generate Report
    
    Args:
        pdf_path: Path to PDF file
        model_path: Path to trained model
        threshold: Classification threshold
        save_report: Save detailed report to file
        output_dir: Directory to save reports
        
    Returns:
        Classification results dictionary
    """
    
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Step 1: Extract text
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(pdf_text)} characters")
    
    # Step 2: Classify causality
    results = classify_causality(
        pdf_text=pdf_text,
        model_path=model_path,
        threshold=threshold,
        verbose=True
    )
    
    # Step 3: Add PDF metadata
    results['pdf_file'] = str(Path(pdf_path).name)
    results['pdf_path'] = str(Path(pdf_path).absolute())
    
    # Step 4: Save report if requested
    if save_report:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report_filename = f"{Path(pdf_path).stem}_causality_report.json"
        report_path = Path(output_dir) / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Report saved: {report_path}")
    
    return results


def process_multiple_pdfs(
    pdf_paths: List[str],
    model_path: str = './models/production_model_final',
    threshold: float = 0.5,
    save_reports: bool = False,
    output_dir: str = './results'
) -> List[Dict]:
    """
    Process multiple PDF files in batch
    
    Args:
        pdf_paths: List of PDF file paths
        model_path: Path to trained model
        threshold: Classification threshold
        save_reports: Save individual reports
        output_dir: Directory to save reports
        
    Returns:
        List of results for each PDF
    """
    
    print(f"\n{'='*70}")
    print(f"BATCH PDF PROCESSING")
    print(f"{'='*70}")
    print(f"Total PDFs: {len(pdf_paths)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"\n[{i}/{len(pdf_paths)}] Processing: {pdf_path}")
        
        try:
            results = process_pdf_file(
                pdf_path=pdf_path,
                model_path=model_path,
                threshold=threshold,
                save_report=save_reports,
                output_dir=output_dir
            )
            all_results.append(results)
            print(f"✓ Success: {results['final_classification']}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            all_results.append({
                'pdf_file': str(Path(pdf_path).name),
                'pdf_path': str(pdf_path),
                'error': str(e),
                'final_classification': 'error'
            })
    
    # Generate summary
    successful = len([r for r in all_results if 'error' not in r])
    related = len([r for r in all_results if r.get('final_classification') == 'related'])
    not_related = len([r for r in all_results if r.get('final_classification') == 'not related'])
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total PDFs: {len(pdf_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(pdf_paths) - successful}")
    print(f"Related: {related}")
    print(f"Not Related: {not_related}")
    print(f"{'='*70}\n")
    
    # Save batch summary
    if save_reports:
        summary_path = Path(output_dir) / 'batch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_pdfs': len(pdf_paths),
                'successful': successful,
                'failed': len(pdf_paths) - successful,
                'related_count': related,
                'not_related_count': not_related,
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Batch summary saved: {summary_path}\n")
    
    return all_results


# Example usage
if __name__ == "__main__":
    # Test the classifier
    print("Testing Drug Causality Classifier...")
    
    classifier = CausalityClassifier()
    
    test_text = "Patient developed severe nausea after taking Drug X"
    result = classifier.predict(test_text, return_probs=True)
    
    print(f"\nTest Text: {test_text}")
    print(f"Classification: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
