"""
Comprehensive verification script to analyze PDFs and verify all drug-event causality sentences
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import json
from inference import CausalityClassifier, extract_text_from_pdf, safe_sent_tokenize
import os
from datetime import datetime

# Configuration
MODEL_PATH = './models/production_model_final'
THRESHOLD = 0.5
PDF_DIR = r'C:\Users\koreo\Downloads\Windsurf files\Articles'
RESULTS_DIR = './results'

print("=" * 100)
print("COMPREHENSIVE DRUG CAUSALITY VERIFICATION")
print("=" * 100)

# Load model
print("\nLoading model...")
classifier = CausalityClassifier(model_path=MODEL_PATH, threshold=THRESHOLD)
print("✓ Model loaded successfully!")

# Get PDF files
pdf_files = [
    Path(PDF_DIR) / 'fphar-16-1498191.pdf',
    Path(PDF_DIR) / 'zh801708001593.pdf'
]

# Process each PDF
all_results = {}

for pdf_path in pdf_files:
    if not pdf_path.exists():
        print(f"\n⚠ File not found: {pdf_path}")
        continue
    
    print(f"\n{'=' * 100}")
    print(f"ANALYZING: {pdf_path.name}")
    print("=" * 100)
    
    # Extract text
    print("\nExtracting text from PDF...")
    pdf_text = extract_text_from_pdf(str(pdf_path))
    print(f"✓ Extracted {len(pdf_text)} characters")
    
    # Tokenize sentences
    print("\nTokenizing sentences...")
    sentences = safe_sent_tokenize(pdf_text)
    print(f"✓ Found {len(sentences)} sentences")
    
    # Analyze each sentence
    print("\nAnalyzing sentences for drug-event causality...")
    related_sentences = []
    not_related_sentences = []
    
    for i, sent in enumerate(sentences, 1):
        if not sent.strip():
            continue
        
        # Get prediction
        result = classifier.predict(sent, return_probs=True)
        
        sentence_info = {
            'sentence_number': i,
            'sentence': sent,
            'prediction': result['prediction'],
            'label': result['label'],
            'confidence': result['confidence'],
            'probability_not_related': result['probabilities']['not_related'],
            'probability_related': result['probabilities']['related']
        }
        
        if result['label'] == 1:  # Related
            related_sentences.append(sentence_info)
        else:
            not_related_sentences.append(sentence_info)
        
        # Progress indicator
        if i % 50 == 0:
            print(f"  Processed {i}/{len(sentences)} sentences...")
    
    # Sort by probability
    related_sentences.sort(key=lambda x: x['probability_related'], reverse=True)
    
    # Store results
    all_results[pdf_path.name] = {
        'pdf_file': pdf_path.name,
        'total_sentences': len(sentences),
        'related_sentences_count': len(related_sentences),
        'not_related_sentences_count': len(not_related_sentences),
        'confidence_score': len(related_sentences) / len(sentences) if sentences else 0,
        'final_classification': 'related' if len(related_sentences) > 0 else 'not related',
        'all_related_sentences': related_sentences,
        'threshold_used': THRESHOLD
    }
    
    # Display summary
    print(f"\n{'=' * 100}")
    print(f"SUMMARY FOR: {pdf_path.name}")
    print("=" * 100)
    print(f"Total Sentences: {len(sentences)}")
    print(f"Related Sentences: {len(related_sentences)}")
    print(f"Not Related Sentences: {len(not_related_sentences)}")
    print(f"Confidence Score: {len(related_sentences) / len(sentences):.4f}")
    print(f"Final Classification: {all_results[pdf_path.name]['final_classification'].upper()}")
    
    # Show top 10 related sentences
    print(f"\n{'=' * 100}")
    print(f"TOP 10 DRUG-EVENT CAUSALITY SENTENCES")
    print("=" * 100)
    
    for i, sent_info in enumerate(related_sentences[:10], 1):
        print(f"\n{i}. [Sentence #{sent_info['sentence_number']}]")
        print(f"   Text: {sent_info['sentence'][:200]}{'...' if len(sent_info['sentence']) > 200 else ''}")
        print(f"   Probability (Related): {sent_info['probability_related']:.6f}")
        print(f"   Confidence: {sent_info['confidence']:.6f}")
        print("-" * 100)

# Save comprehensive results
print(f"\n{'=' * 100}")
print("SAVING COMPREHENSIVE RESULTS")
print("=" * 100)

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save detailed results for each PDF
for pdf_name, results in all_results.items():
    filename = Path(pdf_name).stem
    
    # Save full detailed report
    detailed_path = Path(RESULTS_DIR) / f"{filename}_comprehensive_{timestamp}.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved detailed report: {detailed_path}")
    
    # Save top 20 related sentences (summary)
    summary_results = {
        'pdf_file': results['pdf_file'],
        'total_sentences': results['total_sentences'],
        'related_sentences_count': results['related_sentences_count'],
        'confidence_score': results['confidence_score'],
        'final_classification': results['final_classification'],
        'top_20_related_sentences': results['all_related_sentences'][:20],
        'threshold_used': results['threshold_used']
    }
    
    summary_path = Path(RESULTS_DIR) / f"{filename}_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary report: {summary_path}")

# Create comparison report
print(f"\n{'=' * 100}")
print("COMPARISON WITH EXISTING RESULTS")
print("=" * 100)

comparison_report = []

for pdf_name, new_results in all_results.items():
    filename = Path(pdf_name).stem
    old_result_path = Path(RESULTS_DIR) / f"{filename}_report.json"
    
    if old_result_path.exists():
        with open(old_result_path, 'r') as f:
            old_results = json.load(f)
        
        comparison = {
            'pdf_file': pdf_name,
            'old_related_count': old_results.get('related_sentences', 0),
            'new_related_count': new_results['related_sentences_count'],
            'old_total_sentences': old_results.get('total_sentences', 0),
            'new_total_sentences': new_results['total_sentences'],
            'difference': new_results['related_sentences_count'] - old_results.get('related_sentences', 0),
            'old_confidence': old_results.get('confidence_score', 0),
            'new_confidence': new_results['confidence_score']
        }
        
        comparison_report.append(comparison)
        
        print(f"\n{pdf_name}:")
        print(f"  Old Results: {comparison['old_related_count']}/{comparison['old_total_sentences']} related sentences")
        print(f"  New Results: {comparison['new_related_count']}/{comparison['new_total_sentences']} related sentences")
        print(f"  Difference: {comparison['difference']:+d} sentences")
        print(f"  Old Confidence: {comparison['old_confidence']:.4f}")
        print(f"  New Confidence: {comparison['new_confidence']:.4f}")
        
        if comparison['difference'] > 0:
            print(f"  ✓ Found {comparison['difference']} additional related sentences!")
        elif comparison['difference'] < 0:
            print(f"  ⚠ Found {abs(comparison['difference'])} fewer related sentences")
        else:
            print(f"  ✓ Same number of related sentences")
    else:
        print(f"\n{pdf_name}: No previous results found - this is a new analysis")
        comparison_report.append({
            'pdf_file': pdf_name,
            'status': 'new_analysis',
            'new_related_count': new_results['related_sentences_count'],
            'new_total_sentences': new_results['total_sentences']
        })

# Save comparison report
comparison_path = Path(RESULTS_DIR) / f"comparison_report_{timestamp}.json"
with open(comparison_path, 'w', encoding='utf-8') as f:
    json.dump(comparison_report, f, indent=2)
print(f"\n✓ Saved comparison report: {comparison_path}")

# Create human-readable verification report
print(f"\n{'=' * 100}")
print("CREATING VERIFICATION REPORT")
print("=" * 100)

report_path = Path(RESULTS_DIR) / f"verification_report_{timestamp}.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DRUG CAUSALITY VERIFICATION REPORT\n")
    f.write("=" * 100 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Threshold: {THRESHOLD}\n\n")
    
    for pdf_name, results in all_results.items():
        f.write("=" * 100 + "\n")
        f.write(f"PDF: {pdf_name}\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total Sentences: {results['total_sentences']}\n")
        f.write(f"Related Sentences: {results['related_sentences_count']}\n")
        f.write(f"Confidence Score: {results['confidence_score']:.4f}\n")
        f.write(f"Classification: {results['final_classification'].upper()}\n\n")
        
        f.write("ALL DRUG-EVENT CAUSALITY RELATED SENTENCES:\n")
        f.write("-" * 100 + "\n\n")
        
        for i, sent_info in enumerate(results['all_related_sentences'], 1):
            f.write(f"{i}. [Sentence #{sent_info['sentence_number']}]\n")
            f.write(f"   {sent_info['sentence']}\n")
            f.write(f"   Probability: {sent_info['probability_related']:.6f}\n")
            f.write(f"   Confidence: {sent_info['confidence']:.6f}\n\n")
        
        f.write("\n")

print(f"✓ Saved verification report: {report_path}")

print(f"\n{'=' * 100}")
print("VERIFICATION COMPLETE!")
print("=" * 100)
print(f"\nGenerated files:")
print(f"  - Detailed JSON reports (with ALL sentences)")
print(f"  - Summary JSON reports (top 20 sentences)")
print(f"  - Comparison report")
print(f"  - Human-readable verification report")
print(f"\nAll files saved to: {RESULTS_DIR}")
print("=" * 100)
