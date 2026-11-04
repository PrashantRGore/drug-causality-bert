# 💊 Drug Causality BERT v1.0

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/PrashantRGore/drug-causality-bert-model)
[![ADE Corpus V2 Benchmark](https://img.shields.io/badge/ADE--V2%20Accuracy-97.59%25-brightgreen)](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification)
[![CI Tests](https://github.com/PrashantRGore/drug-causality-bert/actions/workflows/ci.yml/badge.svg)](https://github.com/PrashantRGore/drug-causality-bert/actions)
[![Version 2.0 Available](https://img.shields.io/badge/%F0%9F%9A%80%20Upgraded%20Version-Drug%20Causality%20BERT%20v2-purple)](https://github.com/PrashantRGore/drug-causality-bert-v2)

A specialized biomedical NLP system and interactive clinical decision-support application for **automated adverse drug event (ADE) causality assessment**, achieving **97.6% accuracy** on the ADE Corpus V2 benchmark.

> [!TIP]
> **Looking for Version 2.0?**  
> Check out **[Drug Causality BERT v2](https://github.com/PrashantRGore/drug-causality-bert-v2)** for live OpenFDA FAERS integration, automated MedDRA standardization, interactive Naranjo scoring, and regulatory PBRER Section 11 safety report compilation.

---

## 📌 Model & System Overview

Drug Causality BERT v1.0 classifies clinical narratives, medical literature abstracts, and pharmacovigilance reports to determine whether an adverse drug reaction is causally associated with a suspect medication.

- **Base Architecture:** [dmis-lab/biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- **Task:** Binary Sequence Classification (Causal ADE vs. Non-Causal / Control)
- **Benchmark Corpus:** [ADE Corpus V2](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification) (4,271 documents, 5,063 drugs, 6,821 ADE annotations)
- **Hugging Face Model Weights:** [PrashantRGore/drug-causality-bert-model](https://huggingface.co/PrashantRGore/drug-causality-bert-model)

---

## 📊 Benchmark Performance (ADE Corpus V2)

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **97.59%** | Robust discrimination between adverse reactions and indications |
| **F1-Score** | **97.59%** | Balanced precision and recall across sparse medical events |
| **Sensitivity (Recall)** | **98.68%** | High safety screening sensitivity to minimize missed signals |
| **Specificity** | **96.50%** | Low false alarm rate on background disease and comorbidities |

---

## ✨ Key Features

- **Single Statement Screening:** Real-time causality inference with calibrated probability progress bars.
- **PDF Document Processing:** Extract and parse medical literature and case report PDFs (supporting modern `pypdf` with `PyPDF2` fallback).
- **Batch Processing:** Concurrent processing of multiple clinical case reports with consolidated JSON exports.
- **Hardware Acceleration:** Auto-detects NVIDIA CUDA GPUs, Apple Silicon MPS, or CPU environments.
- **Dual Model Loading:** Automatically uses local model weights if present, with transparent fallback to Hugging Face Model Hub.
- **Configurable Threshold:** Interactive sensitivity/specificity slider for strict regulatory vs. exploratory screening.

---

## 🚀 Quick Start & Installation

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/PrashantRGore/drug-causality-bert.git
cd drug-causality-bert

# Install dependencies
pip install -r requirements.txt
```

### 2. Python Inference

```python
from src.inference import CausalityClassifier

# Automatically resolves local weights or downloads from Hugging Face Hub
classifier = CausalityClassifier()

statement = "Patient developed acute anaphylaxis and severe rash within 20 minutes of amoxicillin administration."
result = classifier.predict(statement, return_probs=True)

print(f"Prediction: {result['prediction'].upper()}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Causal Probability: {result['probabilities']['related']:.2%}")
```

**Output:**
```text
Prediction: RELATED
Confidence: 99.99%
Causal Probability: 99.99%
```

### 3. Launch Interactive Streamlit App

```bash
streamlit run streamlit_app.py
```

The web application provides 3 dedicated operational tabs:
1. **📝 Single Statement:** Fast clinical sentence screening with confidence gauges.
2. **📄 PDF Case Analysis:** Document-level text extraction, sentence ranking, and JSON summary export.
3. **📁 Batch Processing:** Multi-document screening with aggregate metrics and batch report generation.

### 4. Running Automated Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📂 Project Architecture

```text
drug-causality-bert/
├── .github/workflows/
│   └── ci.yml                     # Automated GitHub Actions test workflow
├── app/
│   ├── requirements.txt           # App dependencies
│   └── streamlit_app.py           # Synchronized application entrypoint
├── models/
│   └── production_model_final/    # Model configuration, tokenizer & safetensors
├── src/
│   ├── __init__.py                # Package initialization
│   └── inference.py               # BioBERT causality engine & PDF parser
├── tests/
│   └── test_v1_components.py      # Automated unit test suite (6/6 tests passing)
├── .gitattributes                 # Git LFS attributes
├── .gitignore                     # Git ignore patterns
├── LICENSE                        # Apache 2.0 License
├── pyproject.toml                 # Modern PEP 518/621 package metadata
├── requirements.txt               # Unified project dependencies
├── streamlit_app.py               # Main Streamlit web application
└── README.md                      # Comprehensive project documentation
```

---

## 🔒 Ethical Safeguards & Regulatory Compliance

- **Research & Decision Support:** This software is designed to augment safety scientists, QPPVs, and medical reviewers. Final causality determinations require qualified medical evaluation.
- **Patient Privacy:** The underlying model was trained exclusively on open-access scientific publications (MEDLINE). De-identification of patient records is required prior to processing clinical PDFs.
- **Zero Cost / Open Source:** Built on 100% free open-source tools with zero paid API dependencies.

---

## 📜 Citation

If you utilize this model or codebase in your pharmacovigilance research, please cite:

```bibtex
@misc{gore2025drugcausalityv1,
  author       = {Gore, Prashant R.},
  title        = {Drug Causality BERT: BioBERT for Automated Pharmacovigilance ADE Detection},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/PrashantRGore/drug-causality-bert}}
}

@article{gurulingappa2012ade,
  title        = {Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports},
  author       = {Gurulingappa, Harsha and Rajput, Abdul Mateen and Roberts, Angus and Fluck, Juliane and Hofmann-Apitius, Martin and Toldo, Luca},
  journal      = {Journal of Biomedical Informatics},
  volume       = {45},
  number       = {5},
  pages        = {885--892},
  year         = {2012},
  publisher    = {Elsevier}
}
```

---

## 📄 License

Distributed under the **Apache-2.0 License**. See [`LICENSE`](LICENSE) for full details.

---

## 👤 Author & Support

- **Author:** Prashant R. Gore
- **GitHub:** [@PrashantRGore](https://github.com/PrashantRGore)
- **LinkedIn:** [linkedin.com/in/prashantgorepg](https://linkedin.com/in/prashantgorepg)
- **Hugging Face:** [@PrashantRGore](https://huggingface.co/PrashantRGore)
- **Issue Tracker:** [Report an Issue](https://github.com/PrashantRGore/drug-causality-bert/issues)
