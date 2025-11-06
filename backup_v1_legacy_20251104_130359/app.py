"""
Drug Causality BERT - Version 2.0
Streamlit Web Application
PBRER/PSUR-Compliant Causality Assessment
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import os
import json
import tempfile
from datetime import datetime
from regulatory_causality_report import create_regulatory_report

# Page configuration
st.set_page_config(
    page_title="Drug Causality BERT v2.0",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💊 Drug Causality BERT v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PBRER/PSUR-Compliant Pharmacovigilance Analysis System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pill.png", width=80)
    st.title("About")
    st.info("""
    **Drug Causality BERT v2.0** is an AI-powered system for automated drug-event causality assessment.
    
    **Features:**
    - PBRER/PSUR-compliant reports
    - WHO-UMC causality categories
    - Naranjo ADR probability scale
    - FDA/EMA regulatory context
    - Professional medical language
    
    **Model Performance:**
    - F1 Score: 0.9759
    - Accuracy: 0.9759
    - Sensitivity: 0.9868
    - Specificity: 0.9650
    """)
    
    st.markdown("---")
    st.subheader("Regulatory Compliance")
    st.markdown("""
    - ICH E2C(R2)
    - FDA 21 CFR
    - EMA GVP Module IX
    - WHO-UMC Guidelines
    """)
    
    st.markdown("---")
    st.caption("Version 2.0 | 2025")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Analyze", "📊 Results", "📋 PBRER Section 11", "ℹ️ Information"])

with tab1:
    st.header("Upload PDF Document for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>📋 Supported Documents:</strong><br>
        • Clinical trial reports<br>
        • Case reports<br>
        • Literature articles<br>
        • Pharmacovigilance documents<br>
        • PBRER/PSUR reports
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document containing drug-event information"
        )
        
        if uploaded_file is not None:
            st.success(f"✓ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.session_state['uploaded_file_path'] = tmp_file_path
            st.session_state['uploaded_file_name'] = uploaded_file.name
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <strong>Analysis Includes:</strong><br><br>
        ✓ Drug identification<br>
        ✓ Adverse event detection<br>
        ✓ Causality assessment<br>
        ✓ WHO-UMC categories<br>
        ✓ Naranjo scores<br>
        ✓ Regulatory context<br>
        ✓ Clinical significance<br>
        ✓ Recommendations
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis button
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔬 Generate Drug Causality Assessment Report",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            with st.spinner("🔄 Analyzing document... This may take 2-5 minutes..."):
                try:
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading BioBERT model...")
                    progress_bar.progress(10)
                    
                    status_text.text("Extracting text from PDF...")
                    progress_bar.progress(30)
                    
                    status_text.text("Performing causality assessment...")
                    progress_bar.progress(50)
                    
                    # Generate report
                    doc_path, json_path = create_regulatory_report(
                        st.session_state['uploaded_file_path']
                    )
                    
                    status_text.text("Generating PBRER-compliant report...")
                    progress_bar.progress(80)
                    
                    # Load results
                    with open(json_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    st.session_state['results'] = results
                    st.session_state['doc_path'] = doc_path
                    st.session_state['json_path'] = json_path
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Analysis complete!")
                    
                    st.balloons()
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Analysis Complete!</h3>
                    <p><strong>Document:</strong> {uploaded_file.name}</p>
                    <p><strong>Report Generated:</strong> {Path(doc_path).name}</p>
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open(doc_path, 'rb') as f:
                            st.download_button(
                                label="📄 Download Word Report",
                                data=f,
                                file_name=Path(doc_path).name,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                    
                    with col2:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="📊 Download JSON Summary",
                                data=f,
                                file_name=Path(json_path).name,
                                mime="application/json",
                                use_container_width=True
                            )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("Analysis Results")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Summary metrics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Sentences",
                value=f"{results['total_sentences']:,}"
            )
        
        with col2:
            st.metric(
                label="Unique Drugs",
                value=results['total_drugs']
            )
        
        with col3:
            st.metric(
                label="Unique Events",
                value=results['total_events']
            )
        
        with col4:
            related_count = sum(d['related_count'] for d in results.get('drug_statistics', []))
            st.metric(
                label="Related Statements",
                value=related_count
            )
        
        st.markdown("---")
        
        # Drug statistics
        if results.get('drug_statistics'):
            st.subheader("🔬 Identified Medicinal Products with Safety Signals")
            
            # Create DataFrame for better display
            import pandas as pd
            
            df_drugs = pd.DataFrame(results['drug_statistics'])
            df_drugs['confidence_pct'] = df_drugs['max_confidence'] * 100
            df_drugs = df_drugs.sort_values('confidence_pct', ascending=False)
            
            # Display top drugs
            st.dataframe(
                df_drugs[['drug', 'related_count', 'confidence_pct']].rename(columns={
                    'drug': 'Drug Name',
                    'related_count': 'Related Statements',
                    'confidence_pct': 'Confidence (%)'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            st.subheader("📈 Confidence Distribution")
            
            import plotly.express as px
            
            fig = px.bar(
                df_drugs.head(10),
                x='drug',
                y='confidence_pct',
                title='Top 10 Drugs by Confidence Score',
                labels={'drug': 'Drug Name', 'confidence_pct': 'Confidence (%)'},
                color='confidence_pct',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Identified drugs and events
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💊 All Identified Drugs")
            for drug in sorted(results['drugs_identified']):
                st.markdown(f"• {drug}")
        
        with col2:
            st.subheader("⚠️ All Identified Events")
            for event in sorted(results['events_identified']):
                st.markdown(f"• {event}")
        
        st.markdown("---")
        
        # Model performance
        st.subheader("🎯 Model Performance Metrics")
        
        perf = results['model_performance']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F1 Score", f"{perf['f1_score']:.4f}")
        with col2:
            st.metric("Accuracy", f"{perf['accuracy']:.4f}")
        with col3:
            st.metric("Sensitivity", f"{perf['sensitivity']:.4f}")
        with col4:
            st.metric("Specificity", f"{perf['specificity']:.4f}")
        
    else:
        st.info("👆 Upload and analyze a PDF document to see results here.")

with tab3:
    st.header("📋 PBRER Section 11 - Company Comment Generator")
    
    st.markdown("""
    <div class="info-box">
    <strong>📝 PBRER Section 11 Format</strong><br>
    Generate drug-specific summaries with company interpretation following ICH E2C(R2) Section 11 guidelines.
    This includes comprehensive analysis of confounding factors, concomitant medications, and temporal relationships.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if analysis has been done
    if 'results' in st.session_state and st.session_state['results'].get('drugs_identified'):
        
        st.subheader("Step 1: Select Drug for Detailed Analysis")
        
        # Get list of identified drugs
        drugs_list = sorted(st.session_state['results']['drugs_identified'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_drug = st.selectbox(
                "Select a drug from the identified list:",
                options=drugs_list,
                help="Choose the drug you want to generate a PBRER Section 11 report for"
            )
        
        with col2:
            st.metric(
                label="Total Drugs Identified",
                value=len(drugs_list)
            )
        
        st.markdown("---")
        
        # Generate button
        if st.button("🔬 Generate PBRER Section 11 Report", type="primary", use_container_width=True):
            
            with st.spinner(f"Generating comprehensive PBRER Section 11 analysis for {selected_drug}..."):
                try:
                    from pbrer_section11_generator import generate_pbrer_section11
                    
                    # Generate report
                    report_text, analysis_results = generate_pbrer_section11(
                        st.session_state['uploaded_file_path'],
                        selected_drug
                    )
                    
                    # Store in session state
                    st.session_state['pbrer_report'] = report_text
                    st.session_state['pbrer_analysis'] = analysis_results
                    st.session_state['pbrer_drug'] = selected_drug
                    
                    st.success(f"✅ PBRER Section 11 report generated for {selected_drug}!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.exception(e)
        
        # Display results if available
        if 'pbrer_report' in st.session_state:
            st.markdown("---")
            st.subheader(f"PBRER Section 11 Report: {st.session_state['pbrer_drug']}")
            
            # Display report in expandable sections
            with st.expander("📄 Complete Report", expanded=True):
                st.text(st.session_state['pbrer_report'])
            
            # Download button
            st.download_button(
                label="📥 Download PBRER Section 11 Report",
                data=st.session_state['pbrer_report'],
                file_name=f"PBRER_Section11_{st.session_state['pbrer_drug']}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Display analysis details
            st.subheader("📊 Detailed Analysis Components")
            
            analysis = st.session_state['pbrer_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Concomitant Medications",
                    len(analysis.get('concomitant_medications', []))
                )
            
            with col2:
                st.metric(
                    "Time to Onset References",
                    len(analysis.get('time_to_onset', []))
                )
            
            with col3:
                st.metric(
                    "Concurrent Conditions",
                    len(analysis.get('concurrent_conditions', []))
                )
            
            # Detailed breakdowns
            col1, col2 = st.columns(2)
            
            with col1:
                if analysis.get('concomitant_medications'):
                    with st.expander("💊 Concomitant Medications Details"):
                        for i, med in enumerate(analysis['concomitant_medications'][:10], 1):
                            st.markdown(f"**{i}.** {med['match']}")
                            st.caption(f"Context: {med['context'][:150]}...")
                            st.markdown("---")
                
                if analysis.get('dose_information'):
                    with st.expander("💉 Dose Information"):
                        for i, dose in enumerate(analysis['dose_information'][:10], 1):
                            st.markdown(f"**{i}.** {dose['match']}")
                            st.caption(f"Context: {dose['context'][:150]}...")
                            st.markdown("---")
            
            with col2:
                if analysis.get('time_to_onset'):
                    with st.expander("⏱️ Time to Onset Information"):
                        for i, time in enumerate(analysis['time_to_onset'][:10], 1):
                            st.markdown(f"**{i}.** {time['match']}")
                            st.caption(f"Context: {time['context'][:150]}...")
                            st.markdown("---")
                
                if analysis.get('concurrent_conditions'):
                    with st.expander("🏥 Concurrent Conditions"):
                        for i, cond in enumerate(analysis['concurrent_conditions'][:10], 1):
                            st.markdown(f"**{i}.** {cond['match']}")
                            st.caption(f"Context: {cond['context'][:150]}...")
                            st.markdown("---")
            
            # Confounding factors
            if analysis.get('confounding_factors'):
                with st.expander("⚠️ Confounding Factors", expanded=True):
                    st.warning(f"**{len(analysis['confounding_factors'])} potential confounding factor(s) identified**")
                    for i, factor in enumerate(analysis['confounding_factors'][:5], 1):
                        st.markdown(f"**{i}.** {factor['sentence'][:300]}...")
                        st.caption(f"Keyword: {factor['keyword']}")
                        st.markdown("---")
    
    else:
        st.info("👆 Please upload and analyze a PDF document first (in the 'Upload & Analyze' tab) to use this feature.")
        
        st.markdown("---")
        
        st.subheader("What is PBRER Section 11?")
        st.markdown("""
        **PBRER Section 11** (Periodic Benefit-Risk Evaluation Report) requires a comprehensive 
        summary and company interpretation of individual case reports and literature articles.
        
        **This tool generates:**
        
        1. **Document Summary** - Focused on the selected drug
        2. **Contextual Information:**
           - Concomitant medications
           - Time to onset
           - Concurrent conditions
           - Dose information
           - Patient demographics
           - Clinical outcomes
        
        3. **Confounding Factors Analysis:**
           - Alternative explanations
           - Potential biases
           - Missing information
        
        4. **Company Comment** in standard PBRER format:
           - Summary of case/literature
           - Company causality assessment
           - Rationale for assessment
           - Confounding factors
           - Conclusion and recommendations
        
        **Regulatory Compliance:**
        - ICH E2C(R2) Section 11 format
        - FDA and EMA guidelines
        - Professional medical language
        - Standardized causality assessment
        """)

with tab4:
    st.header("ℹ️ System Information")
    
    st.subheader("📋 What is Drug Causality Assessment?")
    st.markdown("""
    Drug causality assessment is the systematic evaluation of the relationship between 
    drug administration and the occurrence of adverse events. This system uses advanced 
    machine learning (BioBERT) to automatically identify and assess potential drug-event 
    associations in medical literature and pharmacovigilance documents.
    """)
    
    st.markdown("---")
    
    st.subheader("🏥 Causality Assessment Scales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **WHO-UMC Causality Categories:**
        
        1. **Certain/Definite** - Clear evidence, no alternatives
        2. **Probable/Likely** - Strong evidence, unlikely other causes
        3. **Possible** - Reasonable evidence, other factors possible
        4. **Unlikely** - Temporal relationship but other causes likely
        5. **Conditional** - More data needed
        6. **Unassessable** - Cannot be judged
        """)
    
    with col2:
        st.markdown("""
        **Naranjo ADR Probability Scale:**
        
        - **Definite (≥9 points)** - Highly probable ADR
        - **Probable (5-8 points)** - Likely ADR
        - **Possible (1-4 points)** - Could be ADR
        - **Doubtful (≤0 points)** - Unlikely ADR
        
        *Based on 10 standardized questions*
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Regulatory Framework")
    st.markdown("""
    This system follows international regulatory guidelines:
    
    - **ICH E2C(R2)** - Periodic Benefit-Risk Evaluation Report (PBRER)
    - **FDA 21 CFR** - Code of Federal Regulations
    - **EMA GVP Module IX** - Signal Management
    - **WHO-UMC** - Causality Assessment Guidelines
    """)
    
    st.markdown("---")
    
    st.subheader("🔬 How It Works")
    st.markdown("""
    1. **Text Extraction** - Extracts text from uploaded PDF
    2. **Sentence Tokenization** - Breaks document into individual sentences
    3. **Drug & Event Detection** - Identifies drug names and adverse events
    4. **Causality Classification** - BioBERT model predicts causality (related/not related)
    5. **WHO-UMC Assessment** - Assigns causality category based on confidence
    6. **Naranjo Scoring** - Calculates ADR probability score
    7. **Report Generation** - Creates professional PBRER-compliant Word document
    """)
    
    st.markdown("---")
    
    st.subheader("📖 Report Contents")
    st.markdown("""
    The generated Word document includes:
    
    - **Executive Summary** - Overview of findings
    - **Signal Detection Summary** - Methodology and approach
    - **Medicinal Products List** - All drugs with safety signals
    - **Detailed Causality Assessment** - Drug-by-drug analysis with:
      - Source statements from document sections
      - Classification and confidence scores
      - WHO-UMC and Naranjo assessments
      - Professional causality statements
    - **Regulatory Assessment** - FDA/EMA context and recommendations
    - **Conclusion** - Overall benefit-risk assessment
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    - Upload clear, text-based PDF documents (not scanned images)
    - Clinical trial reports and case studies work best
    - Ensure documents contain drug names and adverse event descriptions
    - Longer documents provide more comprehensive analysis
    - Review the Word report for detailed findings
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Drug Causality BERT v2.0</strong></p>
    <p>PBRER/PSUR-Compliant Pharmacovigilance Analysis System</p>
    <p>© 2025 | Powered by BioBERT</p>
</div>
""", unsafe_allow_html=True)
