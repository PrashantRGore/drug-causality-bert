import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import nltk

nltk.download('punkt')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOW THIS IMPORT WILL WORK!
from src.inference import (
    CausalityClassifier, 
    extract_text_from_pdf, 
    classify_causality, 
    process_pdf_file, 
    process_multiple_pdfs
)

# App Configuration
st.set_page_config(
    page_title="Drug Causality Classifier",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("💊 Drug Causality Classifier")
st.caption("BioBERT Model | F1 Score: 97.59% | Sensitivity: 98.68% | Specificity: 96.50%")

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        return CausalityClassifier("models/production_model_final")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

classifier = load_model()

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider(
    "Classification Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Higher threshold = stricter causality detection"
)

st.sidebar.info(
    "**Threshold Guide:**\n"
    "- 0.3-0.4: High sensitivity (catch all events)\n"
    "- 0.5: Balanced performance\n"
    "- 0.7-0.8: High precision (reduce false alarms)"
)

# Main Content
tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📄 PDF Analysis", "📁 Batch Processing"])

# TAB 1: Single Text Classification
with tab1:
    st.header("📝 Single Statement Classification")
    st.write("Enter medical text to classify drug-adverse event causality:")
    
    text_input = st.text_area(
        "Medical Text:",
        height=150,
        placeholder="e.g., Patient developed severe nausea and vomiting 2 hours after taking Drug X. Clinical assessment confirmed drug-related causality."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Classify Text", type="primary", use_container_width=True):
            if text_input and classifier:
                with st.spinner("Analyzing text..."):
                    result = classifier.predict(text_input, threshold)
                
                # Display Results
                st.subheader("📊 Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    classification = result['prediction'].upper()
                    color = "green" if result['prediction'] == 'related' else "red"
                    st.markdown(f"**Classification:** :{color}[{classification}]")
                
                with result_col2:
                    confidence_pct = result['confidence'] * 100
                    st.metric("Confidence", f"{confidence_pct:.1f}%")
                
                # Probability Distribution
                st.subheader("📈 Probability Distribution")
                probs = result['probabilities']
                
                # Progress bars
                st.write("**Related (Drug-Caused):**")
                st.progress(probs['related'], text=f"{probs['related']:.2%}")
                
                st.write("**Not Related:**")
                st.progress(probs['not_related'], text=f"{probs['not_related']:.2%}")
                
                # Raw JSON Output
                with st.expander("🔍 View Raw Results"):
                    st.json(result)
                    
            elif not classifier:
                st.error("Model not loaded properly.")
            else:
                st.warning("Please enter text to classify.")
    
    with col2:
        st.info(
            "**Example Inputs:**\n\n"
            "**Related:** _Patient developed rash after taking aspirin. Symptoms resolved after discontinuation._\n\n"
            "**Not Related:** _Patient has a history of diabetes and hypertension. Takes metformin daily._"
        )

# TAB 2: PDF Analysis
with tab2:
    st.header("📄 PDF Document Analysis")
    st.write("Upload a PDF document for comprehensive drug-adverse event analysis:")
    
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload medical documents, case reports, or clinical notes"
    )
    
    if pdf_file and classifier:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, pdf_file.name)
        
        with open(temp_path, "wb") as tmp_f:
            tmp_f.write(pdf_file.getbuffer())
        
        # Analysis Button
        if st.button("🔍 Analyze PDF", type="primary", use_container_width=True):
            with st.spinner(f"Processing {pdf_file.name}..."):
                try:
                    # Extract and classify
                    pdf_text = extract_text_from_pdf(temp_path)
                    results = classify_causality(pdf_text, threshold=threshold)
                    
                    # Display Summary
                    st.subheader("📊 Analysis Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        classification = results['final_classification'].upper()
                        color = "green" if results['final_classification'] == 'related' else "red"
                        st.markdown(f"**Overall:** :{color}[{classification}]")
                    
                    with summary_col2:
                        confidence_pct = results['confidence_score'] * 100
                        st.metric("Confidence", f"{confidence_pct:.1f}%")
                    
                    with summary_col3:
                        st.metric("Total Sentences", results['total_sentences'])
                    
                    # Sentence Breakdown
                    st.subheader("🔍 Sentence Analysis")
                    
                    breakdown_col1, breakdown_col2 = st.columns(2)
                    
                    with breakdown_col1:
                        st.metric("Related Sentences", results['related_sentences'])
                    
                    with breakdown_col2:
                        st.metric("Not Related", results['not_related_sentences'])
                    
                    # Top Related Sentences
                    if results['related_sentences'] > 0:
                        st.subheader("🎯 Top Related Sentences")
                        
                        for i, sent_detail in enumerate(results.get('top_related_sentences', []), 1):
                            confidence = sent_detail['probability_related']
                            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                            
                            st.markdown(f"**{i}.** ({confidence:.1%} confidence)")
                            st.markdown(f":{confidence_color}[{sent_detail['sentence']}]")
                            st.write("")
                    
                    # Download Button
                    st.subheader("💾 Download Report")
                    
                    import json
                    report_json = json.dumps(results, indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=report_json,
                        file_name=f"{pdf_file.name}_causality_report.json",
                        mime="application/json"
                    )
                    
                    # Raw Results Expander
                    with st.expander("🔍 View Full Results"):
                        st.json(results)
                        
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.info("Please ensure the PDF contains readable text and try again.")
                
                # Clean up temp file
                finally:
                    try:
                        os.remove(temp_path)
                    except:
                        pass

# TAB 3: Batch Processing
with tab3:
    st.header("📁 Batch PDF Processing")
    st.write("Upload multiple PDF files for batch causality analysis:")
    
    batch_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple medical documents for batch analysis"
    )
    
    if batch_files and classifier:
        st.write(f"**Selected files:** {len(batch_files)} PDFs")
        
        for i, file in enumerate(batch_files, 1):
            st.write(f"{i}. {file.name}")
        
        if st.button("🔍 Process All PDFs", type="primary", use_container_width=True):
            # Create temporary paths for all files
            batch_temp_paths = []
            temp_dir = tempfile.gettempdir()
            
            try:
                # Save all files temporarily
                for batch_file in batch_files:
                    temp_path = os.path.join(temp_dir, batch_file.name)
                    with open(temp_path, "wb") as tmp_f:
                        tmp_f.write(batch_file.getbuffer())
                    batch_temp_paths.append(temp_path)
                
                # Process all files
                with st.spinner(f"Processing {len(batch_files)} files..."):
                    batch_results = process_multiple_pdfs(batch_temp_paths, threshold=threshold)
                
                # Display Batch Summary
                st.subheader("📊 Batch Analysis Summary")
                
                # Overall stats
                total_files = len(batch_results)
                successful = len([r for r in batch_results if 'error' not in r])
                related_count = len([r for r in batch_results if r.get('final_classification') == 'related'])
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.metric("Total Files", total_files)
                
                with stat_col2:
                    st.metric("Successfully Processed", successful)
                
                with stat_col3:
                    st.metric("Drug-Related Files", related_count)
                
                # Individual Results
                st.subheader("📄 Individual Results")
                
                for i, res in enumerate(batch_results, 1):
                    if 'error' in res:
                        st.error(f"**{i}. {res['pdf_file']}:** Error - {res['error']}")
                    else:
                        classification = res['final_classification'].upper()
                        confidence = res.get('confidence_score', 0) * 100
                        color = "green" if res['final_classification'] == 'related' else "red"
                        
                        st.markdown(f"**{i}. {res['pdf_file']}:** :{color}[{classification}] ({confidence:.1f}% confidence)")
                
                # Download Batch Summary
                st.subheader("💾 Download Batch Report")
                
                import json
                batch_report = {
                    'summary': {
                        'total_files': total_files,
                        'successful': successful,
                        'related_count': related_count,
                        'threshold_used': threshold
                    },
                    'individual_results': batch_results
                }
                
                batch_json = json.dumps(batch_report, indent=2)
                
                st.download_button(
                    label="📥 Download Batch Summary",
                    data=batch_json,
                    file_name="batch_causality_summary.json",
                    mime="application/json"
                )
                
                # Raw Results Expander
                with st.expander("🔍 View Full Batch Results"):
                    st.json(batch_results)
                    
            except Exception as e:
                st.error(f"Batch processing error: {str(e)}")
            
            finally:
                # Clean up all temp files
                for temp_path in batch_temp_paths:
                    try:
                        os.remove(temp_path)
                    except:
                        pass

# Footer
st.markdown("---")
st.markdown(
    "**Built with BioBERT for Pharmacovigilance** | "
    "Developed for clinical decision support and regulatory compliance"
)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.markdown(
    "- **F1 Score:** 97.59%\n"
    "- **Accuracy:** 97.59%\n"
    "- **Sensitivity:** 98.68%\n"
    "- **Specificity:** 96.50%"
)

st.sidebar.markdown("### 🏥 Clinical Use")
st.sidebar.markdown(
    "This tool assists in:\n"
    "- Adverse event detection\n"
    "- Pharmacovigilance screening\n"
    "- Clinical report analysis\n"
    "- Regulatory compliance"
)


