# Drug Causality BERT v2.0 🔬💊

**Professional PBRER/PSUR-Compliant Drug-Event Causality Assessment System**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

## 🌟 What's New in v2.0

- ✅ **PBRER/PSUR-Compliant Language** - Professional medical terminology following ICH E2C(R2)
- ✅ **Enhanced Regulatory Context** - FDA/EMA guidelines integration
- ✅ **Streamlit Web Application** - User-friendly interface with drag-and-drop PDF upload
- ✅ **Improved Drug Detection** - Enhanced pattern recognition for drug names
- ✅ **Professional Report Format** - PBRER-style Word documents
- ✅ **Interactive Visualizations** - Charts and metrics in web app
- ✅ **Batch Processing** - Analyze multiple documents
- ✅ **JSON API** - Programmatic access to results

## 📋 Overview

Drug Causality BERT v2.0 is an AI-powered system that automatically analyzes medical literature and pharmacovigilance documents to identify and assess drug-event causality relationships. The system uses a fine-tuned BioBERT model to classify sentences as causality-related or not, following international regulatory guidelines.

### Key Features

- **Automated Causality Assessment** - BioBERT-based classification (F1: 0.9759)
- **WHO-UMC Causality Categories** - 6-level classification system
- **Naranjo ADR Probability Scale** - Standardized scoring (0-13 points)
- **Regulatory Compliance** - ICH E2C(R2), FDA 21 CFR, EMA GVP Module IX
- **Professional Reports** - PBRER/PSUR-style Word documents
- **Web Interface** - Streamlit app for easy access
- **Jupyter Notebooks** - Interactive analysis environment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drug-causality-bert.git
cd drug-causality-bert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_v2.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using Python Script

```python
from regulatory_causality_report import create_regulatory_report

# Generate report
doc_path, json_path = create_regulatory_report('your_document.pdf')

print(f"Word Report: {doc_path}")
print(f"JSON Summary: {json_path}")
```

### Using Jupyter Notebook

```bash
jupyter notebook drug_causality_regulatory.ipynb
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.9759 |
| **Accuracy** | 0.9759 |
| **Sensitivity** | 0.9868 |
| **Specificity** | 0.9650 |

## 🏥 Causality Assessment Scales

### WHO-UMC Causality Categories

1. **Certain/Definite** - Clear evidence, no alternative explanations
2. **Probable/Likely** - Strong evidence, unlikely other causes
3. **Possible** - Reasonable evidence, other factors possible
4. **Unlikely** - Temporal relationship but other causes likely
5. **Conditional/Unclassified** - More data needed
6. **Unassessable/Unclassifiable** - Cannot be judged

### Naranjo ADR Probability Scale

- **Definite (≥9 points)** - Highly probable adverse drug reaction
- **Probable (5-8 points)** - Likely adverse drug reaction
- **Possible (1-4 points)** - Could be adverse drug reaction
- **Doubtful (≤0 points)** - Unlikely adverse drug reaction

## 📁 Project Structure

```
drug-causality-bert/
├── app.py                              # Streamlit web application ⭐
├── regulatory_causality_report.py      # Main report generator
├── pbrer_psur_templates.py            # Professional language templates ⭐
├── drug_causality_regulatory.ipynb    # Jupyter notebook
├── requirements_v2.txt                # Python dependencies ⭐
│
├── src/
│   ├── __init__.py
│   └── inference.py                   # Core inference module
│
├── models/
│   └── production_model_final/        # BioBERT model (437 MB)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── vocab.txt
│
├── data/
│   └── raw/                           # Upload PDFs here
│
├── results/                           # Generated reports
│
└── README_v2.md                       # This file ⭐
```

## 🎯 Use Cases

- **Pharmacovigilance** - Automated signal detection
- **Literature Review** - Systematic causality assessment
- **Clinical Trials** - Adverse event analysis
- **Regulatory Submissions** - PBRER/PSUR preparation
- **Drug Safety** - Post-marketing surveillance
- **Medical Research** - Drug-event relationship studies

## 📖 Generated Report Contents

The Word document includes:

1. **Executive Summary**
   - Document metadata
   - Summary statistics
   - Key findings

2. **Signal Detection Summary**
   - Methodology
   - Model performance
   - Quality metrics

3. **Medicinal Products with Safety Signals**
   - Drug list with confidence scores
   - Associated adverse events
   - Number of related statements

4. **Detailed Causality Assessment** (For each drug)
   - Section-wise analysis (Abstract, Methods, Results, Discussion, Conclusion)
   - Source statements from document
   - Classification and confidence scores
   - WHO-UMC causality categories
   - Naranjo ADR probability scores
   - Professional causality statements

5. **Regulatory Assessment**
   - FDA/EMA guidelines context
   - Clinical significance
   - Recommended actions (RMP, PSUR, labeling)
   - Risk minimization measures

6. **Conclusion**
   - Overall benefit-risk assessment
   - Key findings summary
   - Quality assurance metrics

## 🔧 Configuration

### Adjusting Classification Threshold

Edit `regulatory_causality_report.py`:

```python
THRESHOLD = 0.5  # Change to 0.6, 0.7, etc.
```

### Adding Custom Drug Patterns

Edit `DRUG_PATTERNS` in `regulatory_causality_report.py`:

```python
DRUG_PATTERNS = [
    r'\b([A-Z][a-z]+(?:mab|nib|prazole|statin))\b',
    r'your_custom_pattern',
]
```

### Adding Custom Event Patterns

Edit `EVENT_PATTERNS`:

```python
EVENT_PATTERNS = [
    r'\b(your_event_pattern)\b',
]
```

## 📚 Documentation

- **[Complete Implementation Summary](COMPLETE_IMPLEMENTATION_SUMMARY.md)** - Full system overview
- **[Regulatory Report Guide](REGULATORY_REPORT_GUIDE.md)** - Detailed usage guide
- **[Verification Summary](VERIFICATION_SUMMARY.md)** - Quality assurance results
- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Getting started

## 🌐 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy `app.py`

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_v2.txt .
RUN pip install -r requirements_v2.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:

```bash
docker build -t drug-causality-bert .
docker run -p 8501:8501 drug-causality-bert
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BioBERT** - Pre-trained biomedical language model
- **Hugging Face** - Transformers library
- **Streamlit** - Web application framework
- **ICH** - International Council for Harmonisation guidelines
- **FDA/EMA** - Regulatory guidance documents
- **WHO-UMC** - Causality assessment methodology

## 📞 Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/drug-causality-bert/issues)
- **Email**: your.email@example.com
- **Documentation**: See `/docs` folder

## 🔄 Version History

### Version 2.0 (2025-11-02)
- ✅ Added PBRER/PSUR-compliant language templates
- ✅ Created Streamlit web application
- ✅ Enhanced drug and event detection
- ✅ Improved regulatory context
- ✅ Added interactive visualizations
- ✅ Professional report formatting

### Version 1.0 (2024)
- Initial release
- Basic causality classification
- WHO-UMC and Naranjo assessments
- Word document generation

## 📊 Example Results

### Sample Analysis

**Document**: Drug-induced cataracts study  
**Drugs Identified**: 32  
**Events Identified**: 8  
**Related Sentences**: 19 (4.58%)

**Top Drugs by Confidence**:
1. Adalimumab → Cataracts (99.99%)
2. Interferon Beta-1A → Cataracts (99.99%)
3. Ibrutinib → Cataracts (99.88%)
4. Atorvastatin → Cataracts (99.78%)
5. Zoledronic Acid → Cataracts (99.47%)

## 🎓 Citation

If you use this system in your research, please cite:

```bibtex
@software{drug_causality_bert_v2,
  title={Drug Causality BERT v2.0: PBRER/PSUR-Compliant Causality Assessment},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/drug-causality-bert}
}
```

## ⚠️ Disclaimer

This system is intended for research and pharmacovigilance support purposes. All results should be reviewed by qualified healthcare professionals and regulatory experts. The system does not replace human judgment in causality assessment and regulatory decision-making.

---

**Drug Causality BERT v2.0** | © 2025 | Powered by BioBERT 🔬💊
