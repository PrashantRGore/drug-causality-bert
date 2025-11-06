# Drug Causality BERT v2.0 - Complete Implementation Guide

## 🎉 **VERSION 2.0 IS READY!**

---

## 📋 **What's New in Version 2.0**

### ✅ **Major Enhancements**

1. **PBRER/PSUR-Compliant Language**
   - Professional medical terminology following ICH E2C(R2)
   - Regulatory-grade report formatting
   - FDA/EMA guidelines integration

2. **PBRER Section 11 Generator** ⭐ NEW
   - Drug-specific document summaries
   - Concomitant medications analysis
   - Time to onset extraction
   - Concurrent conditions identification
   - Confounding factors analysis
   - Company causality assessment
   - Regulatory recommendations

3. **Enhanced Streamlit Web Application**
   - Two-step workflow
   - Step 1: Generate Drug Causality Assessment Report
   - Step 2: Generate PBRER Section 11 Company Comment
   - Automatic downloads
   - Interactive visualizations

4. **Improved Drug & Event Detection**
   - Enhanced pattern recognition
   - Better extraction accuracy
   - More drug name patterns

---

## 🚀 **Quick Start Guide**

### **Installation**

```bash
# Navigate to project directory
cd c:\Users\koreo\drug-causality-bert

# Install/update dependencies
pip install -r requirements_v2.txt

# Download NLTK data (if not already done)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### **Running the Application**

```bash
# Run the new v2.0 app
streamlit run app_v2.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 **How to Use the App**

### **Step 1: Upload PDF & Generate Drug Causality Assessment Report**

1. **Upload your PDF document**
   - Click "Browse files" or drag & drop
   - Supported: Clinical trials, case reports, literature articles

2. **Click "Generate Drug Causality Assessment Report"**
   - Wait 2-5 minutes for analysis
   - System will:
     - Extract text from PDF
     - Identify all drug names
     - Detect adverse events
     - Perform causality classification
     - Generate comprehensive Word report

3. **View Results**
   - Summary statistics (drugs, events, sentences)
   - Top identified drugs with confidence scores
   - Download Word report (PBRER-compliant)
   - Download JSON summary (structured data)

### **Step 2: Generate PBRER Section 11 - Company Comment**

1. **Select a drug from the dropdown**
   - All identified drugs are listed
   - Choose the drug you want detailed analysis for

2. **Click "Generate PBRER Section 11 for [Drug Name]"**
   - System will analyze:
     - All sentences mentioning the drug
     - Concomitant medications
     - Time to onset information
     - Concurrent medical conditions
     - Dose information
     - Patient demographics
     - Clinical outcomes
     - Confounding factors

3. **View & Download Report**
   - Complete PBRER Section 11 report displayed
   - Includes:
     - Document summary (drug-specific)
     - Contextual information
     - Confounding factors
     - Company causality assessment
     - Regulatory recommendations
   - Click "Download" to save as text file
   - Report is ready for regulatory submission

---

## 📁 **Generated Files**

### **From Step 1: Drug Causality Assessment**

**Word Document** (`.docx`)
- Location: `./results/`
- Filename: `[PDF_name]_PBRER_report_v2_[timestamp].docx`
- Contents:
  - Executive Summary
  - Signal Detection Summary
  - Medicinal Products with Safety Signals
  - Detailed Causality Assessment (by drug)
  - Regulatory Assessment
  - Conclusion

**JSON Summary** (`.json`)
- Location: `./results/`
- Filename: `[PDF_name]_summary_[timestamp].json`
- Contents:
  - All drugs identified
  - All events identified
  - Drug statistics with confidence scores
  - Model performance metrics

### **From Step 2: PBRER Section 11**

**Text Report** (`.txt`)
- Downloaded directly to your browser's download folder
- Filename: `PBRER_Section11_[DrugName]_[timestamp].txt`
- Contents:
  - Document Summary
  - Contextual Information
    - Concomitant medications
    - Time to onset
    - Concurrent conditions
    - Dose information
    - Patient demographics
    - Clinical outcomes
  - Confounding Factors Analysis
  - Company Comment (PBRER Section 11 format)
    - Summary of literature/case
    - Company causality assessment
    - Rationale
    - Confounding factors
    - Conclusion and recommendations

---

## 🎯 **Use Cases**

### **1. Pharmacovigilance Signal Detection**
- Upload post-marketing surveillance reports
- Identify potential drug-event associations
- Generate regulatory-compliant documentation

### **2. Literature Review for PBRER/PSUR**
- Analyze published case reports
- Extract drug-specific information
- Generate Section 11 company comments

### **3. Clinical Trial Safety Analysis**
- Process clinical trial reports
- Identify adverse events
- Assess causality relationships

### **4. Regulatory Submissions**
- Generate PBRER Section 11 reports
- Create PSUR documentation
- Prepare FDA/EMA submissions

---

## 📊 **Example Workflow**

### **Scenario: Analyzing a Drug-Induced Cataract Study**

1. **Upload PDF**: `drug_induced_cataracts_study.pdf`

2. **Step 1 Results**:
   - **32 drugs identified**: Adalimumab, Interferon Beta-1A, Ibrutinib, etc.
   - **8 adverse events**: Cataracts, glaucoma, visual impairment, etc.
   - **19 causality-related sentences** (4.58% of document)
   - **Word report generated**: Complete analysis with WHO-UMC and Naranjo scores

3. **Step 2: Select "Adalimumab"**
   - **Concomitant medications**: 3 references found
   - **Time to onset**: "within 6 months", "after 12 weeks"
   - **Concurrent conditions**: 2 conditions identified
   - **Confounding factors**: 1 potential confounder
   - **Company assessment**: "RELATED" (confidence: 99.99%)
   - **Recommendation**: Include in PSUR, evaluate labeling update

4. **Downloads**:
   - `drug_induced_cataracts_study_PBRER_report_v2_20251102.docx`
   - `drug_induced_cataracts_study_summary_20251102.json`
   - `PBRER_Section11_Adalimumab_20251102.txt`

---

## 🔧 **Files Created for v2.0**

### **Core Application Files**

1. **`app_v2.py`** ⭐ **MAIN APP**
   - Complete Streamlit application
   - Two-step workflow
   - All features integrated

2. **`pbrer_section11_generator.py`** ⭐ **NEW**
   - PBRER Section 11 report generator
   - Drug-specific analysis
   - Company comment formatter

3. **`pbrer_psur_templates.py`** ⭐ **NEW**
   - Professional medical language templates
   - PBRER/PSUR-compliant terminology
   - Regulatory action recommendations

4. **`regulatory_causality_report.py`**
   - Main report generator (existing, enhanced)
   - Word document creation
   - Causality assessment

### **Configuration Files**

5. **`requirements_v2.txt`**
   - Updated Python dependencies
   - Includes all necessary packages

6. **`.gitignore_v2`**
   - Updated Git ignore rules
   - Excludes large model files
   - Excludes generated reports

### **Documentation Files**

7. **`README_v2.md`**
   - Complete project documentation
   - Installation instructions
   - Usage examples

8. **`GITHUB_DEPLOYMENT_GUIDE.md`**
   - Step-by-step GitHub deployment
   - Streamlit Cloud deployment
   - Docker deployment

9. **`V2_COMPLETE_GUIDE.md`** (this file)
   - Complete implementation guide
   - Usage instructions
   - Examples

10. **`setup_v2.py`**
    - Automated setup script
    - Checks dependencies
    - Creates directories

---

## 🌐 **Deploying to GitHub**

### **Quick Deployment**

```bash
# Initialize git (if not already done)
git init

# Copy new .gitignore
copy .gitignore_v2 .gitignore

# Stage all files
git add .

# Commit
git commit -m "Drug Causality BERT v2.0 - PBRER/PSUR compliant with Section 11 generator"

# Create GitHub repository (on GitHub.com)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/drug-causality-bert-v2.git
git branch -M main
git push -u origin main
```

### **Deploy Streamlit App**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `app_v2.py`
6. Click "Deploy!"

**Your app will be live at**: `https://YOUR_USERNAME-drug-causality-bert-v2.streamlit.app`

---

## 📚 **Regulatory Compliance**

### **ICH E2C(R2) - PBRER**

The system follows ICH E2C(R2) guidelines for Periodic Benefit-Risk Evaluation Reports:

- **Section 11**: Summary of literature and company interpretation
- Professional medical terminology
- Standardized causality assessment
- Regulatory recommendations

### **FDA 21 CFR**

Compliant with FDA Code of Federal Regulations:

- Adverse event reporting standards
- Causality assessment criteria
- Documentation requirements

### **EMA GVP Module IX**

Follows EMA Good Pharmacovigilance Practices:

- Signal management
- Causality assessment
- Risk communication

### **WHO-UMC Guidelines**

Uses WHO-UMC causality assessment categories:

- Certain/Definite
- Probable/Likely
- Possible
- Unlikely
- Conditional
- Unassessable

---

## 🎓 **Model Performance**

| Metric | Score | Description |
|--------|-------|-------------|
| **F1 Score** | 0.9759 | Harmonic mean of precision and recall |
| **Accuracy** | 0.9759 | Overall classification accuracy |
| **Sensitivity** | 0.9868 | True positive rate (recall) |
| **Specificity** | 0.9650 | True negative rate |

**Model**: BioBERT fine-tuned on drug causality data  
**Training Data**: Curated pharmacovigilance corpus  
**Classification**: Binary (related/not related)

---

## 💡 **Tips for Best Results**

### **PDF Quality**

- ✅ Use text-based PDFs (not scanned images)
- ✅ Clear, readable text
- ✅ Proper formatting

### **Document Types**

- ✅ Clinical trial reports
- ✅ Case reports
- ✅ Literature articles
- ✅ Pharmacovigilance documents
- ✅ PBRER/PSUR reports

### **Analysis**

- ✅ Review Word report for full details
- ✅ Check confidence scores
- ✅ Verify drug names are correct
- ✅ Review confounding factors
- ✅ Use PBRER Section 11 for regulatory submissions

---

## 🔍 **Troubleshooting**

### **Issue: Model files not found**

**Solution**: Ensure BioBERT model files are in `models/production_model_final/`

### **Issue: NLTK data not found**

**Solution**: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`

### **Issue: PDF extraction fails**

**Solution**: Ensure PDF is text-based, not scanned image

### **Issue: No drugs identified**

**Solution**: Check if drug names are in standard format (e.g., Adalimumab, not adalimumab)

### **Issue: Streamlit app won't start**

**Solution**: 
```bash
pip install --upgrade streamlit
streamlit run app_v2.py
```

---

## 📞 **Support**

For questions or issues:

1. Check this guide
2. Review `README_v2.md`
3. Check `GITHUB_DEPLOYMENT_GUIDE.md`
4. Create GitHub issue (if repository is public)

---

## ✅ **Version 2.0 Checklist**

- [x] PBRER/PSUR-compliant language templates
- [x] PBRER Section 11 generator
- [x] Drug-specific summary generation
- [x] Concomitant medications extraction
- [x] Time to onset analysis
- [x] Concurrent conditions identification
- [x] Confounding factors analysis
- [x] Company comment formatter
- [x] Enhanced Streamlit app (two-step workflow)
- [x] Automatic download functionality
- [x] Interactive visualizations
- [x] Comprehensive documentation
- [x] GitHub deployment guide
- [x] Setup automation script

---

## 🎉 **You're All Set!**

**To start using Drug Causality BERT v2.0:**

```bash
streamlit run app_v2.py
```

**Then:**

1. Upload your PDF
2. Generate Drug Causality Assessment Report
3. Select a drug
4. Generate PBRER Section 11 Report
5. Download both reports

**Your complete pharmacovigilance analysis system is ready! 🚀**

---

**Drug Causality BERT v2.0** | © 2025 | Powered by BioBERT 🔬💊
