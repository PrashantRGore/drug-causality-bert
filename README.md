# 💊 Drug Causality BERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drug-causality-bert-dqpeogrcst87jerttmlmlq.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

BioBERT-based system for automated drug-adverse event causality assessment with multi-modal analysis capabilities.

## 🚀 Live Demo

**[Try the App](https://drug-causality-bert-dqpeogrcst87jerttmlmlq.streamlit.app/)**

## 📊 Performance Metrics

- **F1 Score**: 97.59%
- **Accuracy**: 97.59%
- **Sensitivity**: 98.68%
- **Specificity**: 96.50%

## ✨ Key Features

- **Single Text Classification**: Instant causality assessment for individual medical statements
- **PDF Document Analysis**: Comprehensive analysis of medical literature and case reports
- **Batch Processing**: Process multiple documents simultaneously
- **Configurable Threshold**: Adjust sensitivity for different use cases
- **Model Performance Metrics**: Real-time display of model accuracy and precision

## 🛠️ Installation

### Prerequisites
- Python 3.11
- Git

### Local Setup

Clone the repository
git clone https://github.com/PrashantRGore/drug-causality-bert.git
cd drug-causality-bert

Install dependencies
pip install -r requirements.txt

Run the application
streamlit run app/streamlit_app.py


## 📁 Project Structure

drug-causality-bert/
├── streamlit_app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── runtime.txt # Python version specification
├── data/ # Dataset files
├── models/ # Model configuration (model hosted on HF)
├── src/
│ ├── init.py
│ └── inference.py # Core inference logic
└── app/
└── streamlit_app.py # Streamlit interface


## 🤗 Model

The BioBERT model is hosted on Hugging Face Hub:
- **Repository**: [PrashantRGore/drug-causality-bert-model](https://huggingface.co/PrashantRGore/drug-causality-bert-model)
- **Base Model**: BioBERT
- **Size**: 438 MB
- **Fine-tuned on**: ADE Corpus V2 dataset

## 📊 Dataset

This model was trained on the **ADE Corpus V2** dataset:

**ADE Corpus V2** (Adverse Drug Event Corpus)
- **Source**: CSIRO (Commonwealth Scientific and Industrial Research Organisation)
- **Content**: Medical case reports annotated for drug-adverse event relationships
- **Size**: 23,516 sentences from PubMed abstracts
- **Annotations**: Binary classification (related/not related)
- **Citation**: Gurulingappa et al. (2012)

### Dataset Reference


## 💻 Usage

### Single Statement Classification


### PDF Document Analysis


### Web Interface

The Streamlit app provides three main features:

1. **Single Text Classification**: Enter medical text for instant causality assessment
2. **PDF Analysis**: Upload PDF documents for comprehensive sentence-by-sentence analysis
3. **Batch Processing**: Process multiple PDFs simultaneously with downloadable reports

## 📦 Dependencies

Key packages:
- `streamlit>=1.28.0`
- `torch>=2.5.0`
- `transformers>=4.35.0`
- `PyPDF2>=3.0.1`
- `nltk>=3.7`
- `pandas`, `numpy`, `scikit-learn`

## 🔧 Configuration

### Threshold Settings
- **0.3-0.4**: High sensitivity (catch all potential events)
- **0.5**: Balanced performance (default, recommended)
- **0.7-0.8**: High precision (reduce false positives)

### Model Architecture
- **Base**: BioBERT (biobert-v1.1)
- **Task**: Binary sequence classification
- **Max Length**: 96 tokens
- **Training**: Fine-tuned on ADE Corpus V2

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Set main file: `app/streamlit_app.py`
4. Deploy

### Local Development
streamlit run app/streamlit_app.py --server.port 8501


## 📝 Citation

If you use this project in your research, please cite:

@software{drug_causality_bert,
author = {Gore, Prashant R.},
title = {Drug Causality BERT: BioBERT-based Causality Assessment System},
year = {2025},
url = {https://github.com/PrashantRGore/drug-causality-bert}
}


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Issues

Found a bug? Please open an issue [here](https://github.com/PrashantRGore/drug-causality-bert/issues).

## 👨‍💻 Author

**Prashant R. Gore**
- GitHub: [@PrashantRGore](https://github.com/PrashantRGore)
- Model: [Hugging Face](https://huggingface.co/PrashantRGore/drug-causality-bert-model)

## 🙏 Acknowledgments

- **ADE Corpus V2**: CSIRO for providing the benchmark dataset for drug-adverse event extraction
- **BioBERT team**: For the pre-trained biomedical language model (Lee et al., 2020)
- **Streamlit**: For the excellent web framework
- **Hugging Face**: For model hosting infrastructure and transformers library

### Key References

1. **Gurulingappa, H., et al. (2012)**. "Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports." *Journal of Biomedical Informatics*, 45(5), 885-892.

2. **Lee, J., et al. (2020)**. "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." *Bioinformatics*, 36(4), 1234-1240.

---

**Note**: This is a research tool for educational and research purposes. Clinical decisions should always be made by qualified healthcare professionals.