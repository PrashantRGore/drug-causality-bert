# 🚀 Drug Causality BERT v2.0 - Quick Reference Card

## ⚡ Launch Command

```bash
streamlit run app_v2.py
```

---

## 📋 Two-Step Workflow

### **Step 1: Drug Causality Assessment**
1. Upload PDF
2. Click "Generate Drug Causality Assessment Report"
3. Download Word report + JSON summary

### **Step 2: PBRER Section 11**
1. Select drug from dropdown
2. Click "Generate PBRER Section 11"
3. Download company comment report

---

## 📁 Files You Need

| File | Purpose |
|------|---------|
| `app_v2.py` | Main application |
| `pbrer_section11_generator.py` | Section 11 generator |
| `pbrer_psur_templates.py` | Language templates |
| `regulatory_causality_report.py` | Report generator |
| `requirements_v2.txt` | Dependencies |

---

## 📥 What You Get

### **From Step 1:**
- ✅ `[filename]_PBRER_report_v2_[timestamp].docx` - Word report
- ✅ `[filename]_summary_[timestamp].json` - JSON data

### **From Step 2:**
- ✅ `PBRER_Section11_[DrugName]_[timestamp].txt` - Company comment

---

## 🔧 Setup (First Time Only)

```bash
pip install -r requirements_v2.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python setup_v2.py
```

---

## 🌐 Deploy to GitHub

```bash
git init
copy .gitignore_v2 .gitignore
git add .
git commit -m "Drug Causality BERT v2.0"
git remote add origin https://github.com/YOUR_USERNAME/drug-causality-bert-v2.git
git push -u origin main
```

---

## 📊 Model Performance

- **F1**: 0.9759
- **Accuracy**: 0.9759
- **Sensitivity**: 0.9868
- **Specificity**: 0.9650

---

## 📚 Documentation

- `V2_COMPLETE_GUIDE.md` - Full guide
- `README_v2.md` - Project overview
- `GITHUB_DEPLOYMENT_GUIDE.md` - Deployment
- `LAUNCH_V2.md` - Launch instructions
- `VERSION_2_SUMMARY.md` - Implementation summary

---

## 💡 Quick Tips

- First analysis takes longer (model loading)
- Large PDFs: 3-5 minutes
- Small PDFs: 1-2 minutes
- Use text-based PDFs (not scanned)
- Review Word report for full details

---

## 🎯 Use Cases

- Pharmacovigilance signal detection
- Literature review for PBRER/PSUR
- Clinical trial safety analysis
- Regulatory submissions (FDA/EMA)

---

## ✅ Ready to Use!

```bash
streamlit run app_v2.py
```

**Access at**: http://localhost:8501

---

**Version 2.0** | **Status**: ✅ Production Ready | **Date**: Nov 2025
