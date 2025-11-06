"""
Quick test script to verify the setup is working correctly
"""
import sys
from pathlib import Path

print("=" * 80)
print("DRUG CAUSALITY BERT - SETUP VERIFICATION")
print("=" * 80)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check if we're in the right directory
current_dir = Path.cwd()
print(f"✓ Current directory: {current_dir}")

# Check for required directories
required_dirs = ['src', 'models', 'data', 'results']
for dir_name in required_dirs:
    dir_path = current_dir / dir_name
    if dir_path.exists():
        print(f"✓ Found directory: {dir_name}/")
    else:
        print(f"⚠ Missing directory: {dir_name}/")

# Check for model files
model_path = current_dir / 'models' / 'production_model_final'
if model_path.exists():
    model_files = list(model_path.glob('*'))
    print(f"✓ Model directory exists with {len(model_files)} files")
else:
    print("⚠ Model directory not found!")

# Check for notebooks
notebooks = list(current_dir.glob('*.ipynb'))
print(f"\n✓ Found {len(notebooks)} notebook(s):")
for nb in notebooks:
    print(f"  - {nb.name}")

# Try importing required packages
print("\nChecking required packages...")
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-learn',
    'nltk': 'NLTK',
    'PyPDF2': 'PyPDF2',
    'ipywidgets': 'IPyWidgets'
}

missing_packages = []
for package, name in packages.items():
    try:
        __import__(package)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        missing_packages.append(package)

# Try loading the model
print("\nTesting model loading...")
try:
    sys.path.insert(0, str(current_dir / 'src'))
    from inference import CausalityClassifier
    
    classifier = CausalityClassifier(
        model_path=str(model_path),
        threshold=0.5
    )
    print("✓ Model loaded successfully!")
    
    # Test prediction
    test_text = "The patient developed rash after taking the medication."
    result = classifier.predict(test_text, return_probs=True)
    print(f"\n✓ Test prediction successful!")
    print(f"  Text: {test_text}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if missing_packages:
    print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
    print("\nTo install missing packages, run:")
    print(f"  pip install {' '.join(missing_packages)}")
else:
    print("\n✅ All packages installed!")

print("\n✅ Setup verification complete!")
print("\nNext steps:")
print("1. Run: jupyter notebook")
print("2. Open: drug_causality_upload.ipynb")
print("3. Start analyzing PDFs!")
print("=" * 80)
