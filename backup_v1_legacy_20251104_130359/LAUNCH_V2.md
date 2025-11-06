# 🚀 Launch Drug Causality BERT v2.0

## ⚡ Quick Launch Commands

### **Option 1: Run the v2.0 App (Recommended)**

```bash
streamlit run app_v2.py
```

### **Option 2: Run Setup First (If First Time)**

```bash
# Run setup to check everything
python setup_v2.py

# Then launch app
streamlit run app_v2.py
```

---

## 📋 What Happens When You Launch

1. **Browser opens automatically** at `http://localhost:8501`
2. **Upload your PDF** document
3. **Click "Generate Drug Causality Assessment Report"**
   - Wait 2-5 minutes
   - Get comprehensive Word report
   - See all identified drugs and events
4. **Select a drug** from the dropdown
5. **Click "Generate PBRER Section 11"**
   - Get drug-specific analysis
   - Download company comment report
   - Ready for regulatory submission

---

## ✅ Pre-Launch Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated (if using)
- [ ] Dependencies installed (`pip install -r requirements_v2.txt`)
- [ ] NLTK data downloaded
- [ ] Model files in `models/production_model_final/`
- [ ] In correct directory (`drug-causality-bert`)

---

## 🎯 First-Time Setup (One-Time Only)

```bash
# 1. Navigate to project directory
cd c:\Users\koreo\drug-causality-bert

# 2. Install dependencies
pip install -r requirements_v2.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Run setup check
python setup_v2.py

# 5. Launch app
streamlit run app_v2.py
```

---

## 🌐 Access the App

Once launched, the app is available at:

- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501 (if accessible on network)

---

## 🛑 Stop the App

Press `Ctrl + C` in the terminal where the app is running

---

## 📱 Features Available in v2.0

### **Step 1: Drug Causality Assessment**
- ✅ PDF upload widget
- ✅ Automatic drug & event detection
- ✅ WHO-UMC causality categories
- ✅ Naranjo ADR probability scores
- ✅ Comprehensive Word report
- ✅ JSON summary export

### **Step 2: PBRER Section 11**
- ✅ Drug selection dropdown
- ✅ Document summary (drug-specific)
- ✅ Concomitant medications
- ✅ Time to onset analysis
- ✅ Concurrent conditions
- ✅ Confounding factors
- ✅ Company causality assessment
- ✅ Regulatory recommendations
- ✅ Automatic download

---

## 💡 Tips

- **First upload**: May take longer as model loads
- **Subsequent analyses**: Faster due to caching
- **Large PDFs**: May take 3-5 minutes
- **Small PDFs**: Usually 1-2 minutes

---

## 🎉 You're Ready!

Just run:

```bash
streamlit run app_v2.py
```

And start analyzing! 🔬💊
