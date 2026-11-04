"""
Drug Causality BERT v1.0 - Interactive Streamlit Application
BioBERT Adverse Drug Event (ADE) Causality Assessment
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    CausalityClassifier,
    extract_text_from_pdf,
    classify_causality,
    process_pdf_file,
    process_multiple_pdfs
)

# Page Configuration
st.set_page_config(
    page_title="Drug Causality BERT v1.0 | Clinical Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title & Subtitle
st.title("💊 Drug Causality BERT v1.0")
st.caption("BioBERT Clinical NLP Model | ADE Corpus V2 Benchmark | F1-Score: 97.59% | Accuracy: 97.59%")

# Cached Model Loader
@st.cache_resource(show_spinner="Loading Drug Causality BioBERT model...")
def load_classifier():
    """Load CausalityClassifier with automatic fallback to local weights or Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        try:
            token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            token = None
    try:
        return CausalityClassifier(hf_token=token)
    except Exception as e:
        st.error(f"Failed to load BioBERT model: {e}")
        return None

classifier = load_classifier()

# Sidebar Configuration & Telemetry
with st.sidebar:
    st.header("⚙️ Configuration")

    if classifier:
        device_label = "CUDA (GPU)" if "cuda" in str(classifier.device) else "MPS (Apple Silicon)" if "mps" in str(classifier.device) else "CPU (Standard)"
        source_label = "Local Weights" if Path(classifier.model_path).exists() else "Hugging Face Hub"
        st.success(f"**BioBERT Model Active**\n- Device: `{device_label}`\n- Source: `{source_label}`")
    else:
        st.warning("⚠️ Model not initialized. Please check connectivity or model weights.")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher threshold = stricter causality requirements (higher precision)"
    )

    st.info(
        "**Threshold Guide:**\n"
        "- `0.3 - 0.4`: High sensitivity (safety screening)\n"
        "- `0.5`: Balanced optimal benchmark cutoff\n"
        "- `0.7 - 0.8`: High specificity (regulatory confirmation)"
    )

    st.markdown("---")
    st.markdown("### 📚 Resources & Links")
    st.markdown("- 🤗 **[Hugging Face Model](https://huggingface.co/PrashantRGore/drug-causality-bert-model)**")
    st.markdown("- 📊 **[Synthetic FAERS 1M Dataset](https://huggingface.co/datasets/PrashantRGore/synthetic-faers-1m-v3)**")
    st.markdown("- 🚀 **[Drug Causality BERT v2](https://github.com/PrashantRGore/drug-causality-bert-v2)**")
    st.markdown("- 🐙 **[GitHub Repository](https://github.com/PrashantRGore/drug-causality-bert)**")

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["📝 Single Statement", "📄 PDF Case Analysis", "📁 Batch PDF Processing"])

# TAB 1: Single Statement Classification
with tab1:
    st.header("📝 Single Statement Causality Screening")
    st.write("Input a clinical narrative statement or adverse event description to assess drug-event causality:")

    default_text = "Patient developed acute urticaria, facial edema, and severe pruritus 30 minutes after taking amoxicillin. Symptoms resolved upon drug discontinuation and antihistamine administration."
    text_input = st.text_area(
        "Clinical Statement:",
        value=default_text,
        height=140,
        placeholder="Enter adverse event description or patient case excerpt..."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔍 Classify Causality", type="primary", use_container_width=True):
            if text_input and classifier:
                with st.spinner("Analyzing clinical statement with BioBERT..."):
                    result = classifier.predict(text_input, threshold_or_return_probs=threshold, return_probs=True)

                st.subheader("📊 Assessment Result")
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    is_related = result["prediction"] == "related"
                    status_text = "CAUSAL ADE (Related)" if is_related else "NON-CAUSAL (Not Related)"
                    status_color = "green" if is_related else "red"
                    st.markdown(f"### Classification: :{status_color}[{status_text}]")

                with res_col2:
                    conf_pct = result["confidence"] * 100
                    st.metric("Model Confidence", f"{conf_pct:.2f}%")

                # Probability Gauge
                st.subheader("📈 Calibrated Probability Distribution")
                probs = result["probabilities"]
                st.write(f"**Causal ADE (Drug-Induced Event):** `{probs['related']:.2%}`")
                st.progress(probs["related"])

                st.write(f"**Non-Causal (Coincidental / Indication):** `{probs['not_related']:.2%}`")
                st.progress(probs["not_related"])

                with st.expander("🔍 View Raw JSON Output"):
                    st.json(result)

            elif not classifier:
                st.error("BioBERT model is not loaded.")
            else:
                st.warning("Please enter a medical statement to classify.")

    with col2:
        st.info(
            "**Benchmark Sample Prompts:**\n\n"
            "**Positive (Causal ADE):**\n"
            "_Patient developed severe hepatotoxicity and elevated AST/ALT 10 days after starting methotrexate therapy._\n\n"
            "**Negative (Indication / Control):**\n"
            "_Patient has a documented history of chronic hypertension and was prescribed lisinopril 10 mg daily._"
        )

# TAB 2: PDF Document Analysis
with tab2:
    st.header("📄 PDF Document Analysis")
    st.write("Upload a clinical case report, MEDLINE abstract, or regulatory CIOMS/MedWatch PDF for automated sentence-level screening:")

    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload clinical case reports or published medical literature (PDF)"
    )

    if pdf_file and classifier:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, pdf_file.name)

        with open(temp_path, "wb") as tmp_f:
            tmp_f.write(pdf_file.getbuffer())

        if st.button("🔍 Analyze PDF Case Report", type="primary", use_container_width=True):
            with st.spinner(f"Extracting and screening sentences from {pdf_file.name}..."):
                try:
                    results = process_pdf_file(temp_path, threshold=threshold)

                    st.subheader("📊 Document Analysis Summary")
                    sum_col1, sum_col2, sum_col3 = st.columns(3)

                    with sum_col1:
                        is_related = results["final_classification"] == "related"
                        color = "green" if is_related else "red"
                        label = "CAUSAL EVIDENCE DETECTED" if is_related else "NO CAUSAL ADE DETECTED"
                        st.markdown(f"**Overall Classification:** :{color}[{label}]")

                    with sum_col2:
                        conf_pct = results["confidence_score"] * 100
                        st.metric("Related Sentence Ratio", f"{conf_pct:.1f}%")

                    with sum_col3:
                        st.metric("Total Sentences Analyzed", results["total_sentences"])

                    st.subheader("🔍 Sentence Breakdown")
                    brk_col1, brk_col2 = st.columns(2)
                    with brk_col1:
                        st.metric("Causal ADE Sentences", results["related_sentences"])
                    with brk_col2:
                        st.metric("Non-Causal Sentences", results["not_related_sentences"])

                    # Top Sentences
                    if results["related_sentences"] > 0:
                        st.subheader("🎯 High-Confidence Causal Sentences")
                        for i, sent_detail in enumerate(results.get("top_related_sentences", []), 1):
                            prob = sent_detail["probability_related"]
                            color = "green" if prob > 0.8 else "orange" if prob > 0.5 else "red"
                            st.markdown(f"**{i}.** ({prob:.1%} probability): :{color}[{sent_detail['sentence']}]")

                    # Download JSON Report
                    st.subheader("💾 Export Findings")
                    report_json = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download JSON Causality Summary",
                        data=report_json,
                        file_name=f"{Path(pdf_file.name).stem}_causality_report.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"Error analyzing PDF: {e}")
                finally:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

# TAB 3: Batch PDF Processing
with tab3:
    st.header("📁 Batch PDF Document Processing")
    st.write("Upload multiple medical documents or clinical notes for automated batch screening:")

    batch_files = st.file_uploader(
        "Select Multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF case files"
    )

    if batch_files and classifier:
        st.write(f"**Selected {len(batch_files)} documents:** " + ", ".join([f.name for f in batch_files[:5]]) + ("..." if len(batch_files) > 5 else ""))

        if st.button("🚀 Process All Documents", type="primary", use_container_width=True):
            batch_temp_paths = []
            temp_dir = tempfile.gettempdir()

            try:
                for b_file in batch_files:
                    t_path = os.path.join(temp_dir, b_file.name)
                    with open(t_path, "wb") as f_out:
                        f_out.write(b_file.getbuffer())
                    batch_temp_paths.append(t_path)

                with st.spinner(f"Processing batch of {len(batch_files)} PDF documents..."):
                    batch_results = process_multiple_pdfs(batch_temp_paths, threshold=threshold)

                st.subheader("📊 Batch Summary")
                total_files = len(batch_results)
                successful = len([r for r in batch_results if "error" not in r])
                related_docs = len([r for r in batch_results if r.get("final_classification") == "related"])

                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    st.metric("Total Processed", total_files)
                with b_col2:
                    st.metric("Successful Extractions", successful)
                with b_col3:
                    st.metric("Causal ADE Positive Cases", related_docs)

                st.subheader("📄 Individual Document Results")
                for i, res in enumerate(batch_results, 1):
                    if "error" in res:
                        st.error(f"**{i}. {res['pdf_file']}**: Error - {res['error']}")
                    else:
                        is_rel = res.get("final_classification") == "related"
                        badge_color = "green" if is_rel else "gray"
                        conf = res.get("confidence_score", 0) * 100
                        st.markdown(f"**{i}. {res['pdf_file']}**: :{badge_color}[{res['final_classification'].upper()}] ({conf:.1f}% related sentences)")

                batch_report_data = {
                    "summary": {
                        "total_documents": total_files,
                        "successful_extractions": successful,
                        "causal_cases": related_docs,
                        "threshold": threshold
                    },
                    "document_details": batch_results
                }

                st.download_button(
                    label="📥 Download Complete Batch Report (JSON)",
                    data=json.dumps(batch_report_data, indent=2),
                    file_name="batch_causality_summary.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Batch processing error: {e}")
            finally:
                for t_path in batch_temp_paths:
                    try:
                        os.remove(t_path)
                    except Exception:
                        pass

# Footer & Disclaimers
st.markdown("---")
st.markdown(
    "**Drug Causality BERT v1.0** | Developed by **Prashant R. Gore** | "
    "Designed for pharmacovigilance research and clinical decision support. "
    "Verify findings with a qualified physician or QPPV."
)
