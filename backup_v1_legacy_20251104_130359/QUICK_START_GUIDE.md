# Quick Start Guide - Drug Causality Analysis

## ✅ New Enhanced Notebook Created!

I've created **`drug_causality_upload.ipynb`** with a file upload widget for easy PDF analysis.

## 🚀 How to Use

### Option 1: Use the Enhanced Notebook (Recommended)

1. **Open Jupyter Notebook**:
   ```powershell
   jupyter notebook
   ```

2. **Open the file**: `drug_causality_upload.ipynb`

3. **Run cells in order**:
   - Cell 1: Install dependencies
   - Cell 2: Import libraries
   - Cell 3: Load the model
   - Cell 4: **Upload PDF widget appears here!**
     - Click "Browse" or drag & drop your PDF
     - Click "Analyze PDF" button
     - Results will appear below

### Option 2: Use the Original Notebook

Open `drug_causality_analysis.ipynb` for more detailed analysis options.

## 📋 Features of the Enhanced Notebook

### 1. **PDF Upload Widget** (Section 3)
- Click and upload PDF files directly
- No need to manually copy files
- Automatic analysis with one click
- Results displayed immediately

### 2. **Custom Text Testing** (Section 4)
- Test individual sentences
- Quick model validation
- Edit the `your_text` variable

### 3. **Batch Processing** (Section 5)
- Process all PDFs in `data/raw/` folder
- Automatic report generation
- Summary statistics

## 📁 File Structure

```
drug-causality-bert/
├── drug_causality_upload.ipynb    ← NEW! Use this one
├── drug_causality_analysis.ipynb  ← Original detailed notebook
├── data/
│   └── raw/                       ← Uploaded PDFs saved here
├── results/                       ← Analysis results saved here
├── models/
│   └── production_model_final/    ← Pre-trained model
└── src/
    └── inference.py               ← Core functions
```

## 🎯 Example Workflow

1. Start Jupyter: `jupyter notebook`
2. Open: `drug_causality_upload.ipynb`
3. Run first 3 cells to load the model
4. In cell 4:
   - Click the upload button
   - Select your PDF file
   - Click "Analyze PDF"
   - View results!
5. Results are automatically saved to `./results/`

## 📊 Output Files

After analysis, you'll find:
- `./results/[filename]_report.json` - Detailed JSON report
- `./results/batch_causality_summary.json` - Batch processing summary

## 💡 Tips

- The upload widget only accepts `.pdf` files
- Uploaded files are saved to `./data/raw/`
- You can process the same file multiple times
- Results include:
  - Classification (related/not related)
  - Confidence score
  - Top related sentences
  - Probability scores

## 🔧 Troubleshooting

**Widget not showing?**
```python
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension
```
Then restart the kernel.

**Model not loading?**
- Ensure you're in the correct directory: `c:\Users\koreo\drug-causality-bert`
- Check that `./models/production_model_final/` exists

## 📞 Need Help?

- Check the original notebook for more detailed examples
- Review the `README.md` for project information
- Examine `./src/inference.py` for implementation details

---

**Happy Analyzing! 🎉**
