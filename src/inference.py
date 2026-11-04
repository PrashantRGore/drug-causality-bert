"""
Drug Causality BERT v1.0 - Core Inference Engine
BioBERT Sequence Classification for Adverse Drug Event (ADE) Causality Detection.
Supports automatic hardware acceleration (CUDA/MPS/CPU) and dual-source model loading.
"""

import os
import re
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modern PDF extraction with pypdf and PyPDF2 fallback
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# Cross-platform secure NLTK tokenizer setup
import nltk

def setup_nltk_safely():
    """Ensure NLTK sentence tokenizers are present without compromising SSL."""
    cache_nltk_dir = os.path.join(tempfile.gettempdir(), "drug_bert_v1_nltk_data")
    if not os.path.exists(cache_nltk_dir):
        try:
            os.makedirs(cache_nltk_dir, exist_ok=True)
        except OSError:
            pass

    if cache_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, cache_nltk_dir)

    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, download_dir=cache_nltk_dir, quiet=True)
            except Exception:
                pass

setup_nltk_safely()


def simple_sentence_tokenize(text: str) -> List[str]:
    """Clean regex sentence boundary detector as zero-dependency fallback."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def safe_sent_tokenize(text: str) -> List[str]:
    """Tokenize sentences using NLTK if available, falling back to regex."""
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return simple_sentence_tokenize(text)


class CausalityClassifier:
    """
    BioBERT Drug-Adverse Event Causality Classifier v1.0
    Fine-tuned on the ADE Corpus V2 benchmark.

    Features:
    - Automatic hardware acceleration (CUDA GPU / Apple Silicon MPS / CPU)
    - Dual loading: auto-detects local model weights with automatic fallback to Hugging Face Hub
    - 512-token sequence support with calibrated softmax probabilities
    """

    HF_MODEL_ID = "PrashantRGore/drug-causality-bert-model"
    LOCAL_FALLBACK_DIR = "./models/production_model_final"

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        hf_token: Optional[str] = None
    ):
        self.threshold = threshold

        # Hardware acceleration detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Determine model path
        self.model_path = model_path or self._resolve_model_path()
        self.token = hf_token or os.environ.get("HF_TOKEN", None)

        # Load Tokenizer & Model
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self) -> str:
        """Check for valid local weights first, else return Hugging Face Hub repo id."""
        local_dir = Path(self.LOCAL_FALLBACK_DIR)
        local_weights = local_dir / "model.safetensors"
        if local_weights.exists() and local_weights.stat().st_size > 1_000_000:
            return str(local_dir)
        return self.HF_MODEL_ID

    def _load_model_and_tokenizer(self):
        """Safely initialize tokenizer and classification head with fallback."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.token,
                local_files_only=Path(self.model_path).exists()
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                token=self.token,
                local_files_only=Path(self.model_path).exists()
            )
            return tokenizer, model
        except Exception:
            # If local path failed, try Hugging Face Hub fallback
            if self.model_path != self.HF_MODEL_ID:
                tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID, token=self.token)
                model = AutoModelForSequenceClassification.from_pretrained(self.HF_MODEL_ID, token=self.token)
                self.model_path = self.HF_MODEL_ID
                return tokenizer, model
            raise

    def predict(
        self,
        text: str,
        threshold_or_return_probs: Union[float, bool, None] = None,
        return_probs: bool = True
    ) -> Dict:
        """
        Predict whether medical text describes a causal adverse drug effect.
        Accepts threshold as float or return_probs as bool flexibly.
        """
        # Parse flexible arguments
        eff_threshold = self.threshold
        eff_return_probs = return_probs
        if isinstance(threshold_or_return_probs, (int, float)):
            eff_threshold = float(threshold_or_return_probs)
        elif isinstance(threshold_or_return_probs, bool):
            eff_return_probs = threshold_or_return_probs

        if not text or not text.strip():
            empty_res = {
                "prediction": "not related",
                "confidence": 0.0,
                "label": 0,
                "device": str(self.device)
            }
            if eff_return_probs:
                empty_res["probabilities"] = {"not_related": 1.0, "related": 0.0}
            return empty_res

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = 1 if probs[1] > eff_threshold else 0

        result = {
            "prediction": "related" if pred == 1 else "not related",
            "confidence": float(probs[pred]),
            "label": int(pred),
            "device": str(self.device)
        }

        if eff_return_probs:
            result["probabilities"] = {
                "not_related": float(probs[0]),
                "related": float(probs[1])
            }

        return result


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract full raw text from a PDF case report using pypdf or PyPDF2."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    text = ""
    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text
        except Exception:
            pass

    if HAS_PYPDF2:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    if not HAS_PYPDF and not HAS_PYPDF2:
        raise ImportError("No PDF parser installed. Please install pypdf: pip install pypdf")

    return text


def classify_causality(
    pdf_text: str,
    model_path: str = "PrashantRGore/drug-causality-bert-model",
    threshold: float = 0.5,
    verbose: bool = False
) -> Dict:
    """
    Process clinical narrative text sentence-by-sentence and compute overall document causality score.
    """
    classifier = CausalityClassifier(model_path=model_path, threshold=threshold)
    sentences = safe_sent_tokenize(pdf_text)

    if verbose:
        print(f"Tokenized {len(sentences)} sentences")

    related_count = 0
    sentence_details = []

    for sent in sentences:
        clean_sent = sent.strip()
        if not clean_sent or len(clean_sent) < 5:
            continue

        result = classifier.predict(clean_sent, return_probs=True)
        if result["label"] == 1:
            related_count += 1
            sentence_details.append({
                "sentence": clean_sent[:150],
                "probability_related": result["probabilities"]["related"],
                "confidence": result["confidence"]
            })

    sentence_details.sort(key=lambda x: x["probability_related"], reverse=True)

    total_valid = max(1, len(sentences))
    not_related_count = max(0, len(sentences) - related_count)

    return {
        "final_classification": "related" if related_count > 0 else "not related",
        "confidence_score": float(related_count / total_valid),
        "related_sentences": related_count,
        "not_related_sentences": not_related_count,
        "total_sentences": len(sentences),
        "top_related_sentences": sentence_details[:5],
        "threshold_used": threshold
    }


def process_pdf_file(
    pdf_path: Union[str, Path],
    model_path: str = "PrashantRGore/drug-causality-bert-model",
    threshold: float = 0.5,
    save_report: bool = False,
    output_dir: str = "./results"
) -> Dict:
    """Extract text from a PDF, run sentence-level causality scoring, and optionally save JSON summary."""
    pdf_text = extract_text_from_pdf(pdf_path)
    results = classify_causality(pdf_text, model_path, threshold)
    results["pdf_file"] = str(Path(pdf_path).name)

    if save_report:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / f"{Path(pdf_path).stem}_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def process_multiple_pdfs(
    pdf_paths: List[Union[str, Path]],
    model_path: str = "PrashantRGore/drug-causality-bert-model",
    threshold: float = 0.5,
    save_reports: bool = False,
    output_dir: str = "./results"
) -> List[Dict]:
    """Process a batch of clinical PDF documents."""
    all_results = []
    for pdf_path in pdf_paths:
        try:
            results = process_pdf_file(pdf_path, model_path, threshold, save_reports, output_dir)
            all_results.append(results)
        except Exception as e:
            all_results.append({
                "pdf_file": str(Path(pdf_path).name),
                "error": str(e),
                "final_classification": "error"
            })
    return all_results
