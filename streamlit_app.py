import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import nltk

nltk.download('punkt')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import CausalityClassifier, extract_text_from_pdf, classify_causality, process_pdf_file, process_multiple_pdfs

st.title("Drug Causality Classifier")
st.caption("BioBERT Model | F1 Score: 97.59%")
model_path = "PrashantRGore/drug-causality-bert-model"
classifier = CausalityClassifier(model_path)
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

st.header("Classify Single Statement")
text = st.text_area("Enter medical text:", height=150)
if st.button("Classify Text"):
    if text:
        result = classifier.predict(text, threshold)
        st.json(result)
    else:
        st.warning("Please enter text.")

st.markdown("---")

st.header("Full PDF Document Analysis")
pdf_file = st.file_uploader("Upload a PDF document:", type=["pdf"])
if pdf_file:
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, pdf_file.name)
    with open(temp_path, "wb") as tmp_f:
        tmp_f.write(pdf_file.getbuffer())
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(temp_path)
        results = classify_causality(pdf_text, threshold=threshold)
    st.subheader("Top 5 Related Sentences")
    for i, sent in enumerate(results.get("top_related_sentences", []), 1):
        st.write(f"{i}. {sent['sentence']} (confidence: {sent['probability_related']:.2%})")
    st.json(results)
    st.download_button(
        label="Download JSON Report",
        data=str(results),
        file_name=f"{pdf_file.name}_causality_report.json"
    )

st.markdown("---")

st.header("Batch PDF Analysis")
batch_files = st.file_uploader("Upload multiple PDFs:", type=["pdf"], accept_multiple_files=True)
if batch_files:
    batch_temp_paths = []
    for batch_file in batch_files:
        temp_path = os.path.join(tempfile.gettempdir(), batch_file.name)
        with open(temp_path, "wb") as tmp_f:
            tmp_f.write(batch_file.getbuffer())
        batch_temp_paths.append(temp_path)
    batch_results = process_multiple_pdfs(batch_temp_paths, threshold=threshold)
    st.subheader("Batch Analysis Summary")
    for res in batch_results:
        st.write(f"{res['pdf_file']}: {res['final_classification'].upper()} (confidence: {res.get('confidence_score', 0):.2%})")
    st.download_button(
        label="Download Batch JSON Summary",
        data=str(batch_results),
        file_name="batch_causality_summary.json"
    )
