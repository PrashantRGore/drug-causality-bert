"""
Drug Causality BERT v2.0 - Setup Script
Prepares the environment and downloads required data
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print('='*60)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        Drug Causality BERT v2.0 - Setup Script          ║
    ║                                                          ║
    ║     PBRER/PSUR-Compliant Causality Assessment System    ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print("\n[1/5] Checking Python version...")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Install requirements
    print("\n[2/5] Installing Python dependencies...")
    if not run_command(
        "pip install -r requirements_v2.txt",
        "Installing packages from requirements_v2.txt"
    ):
        print("\n⚠ Warning: Some packages may have failed to install")
        print("  You can try installing them manually")
    
    # Download NLTK data
    print("\n[3/5] Downloading NLTK data...")
    try:
        import nltk
        print("  Downloading 'punkt'...")
        nltk.download('punkt', quiet=True)
        print("  Downloading 'punkt_tab'...")
        nltk.download('punkt_tab', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
    
    # Create directories
    print("\n[4/5] Creating directory structure...")
    directories = [
        'data/raw',
        'results',
        'models/production_model_final',
        '.streamlit'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    # Check model files
    print("\n[5/5] Checking model files...")
    model_path = 'models/production_model_final'
    required_files = ['config.json', 'tokenizer.json', 'vocab.txt']
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ Found: {file}")
        else:
            print(f"  ✗ Missing: {file}")
            missing_files.append(file)
    
    # Check for model weights
    model_weights = ['model.safetensors', 'pytorch_model.bin']
    has_weights = any(os.path.exists(os.path.join(model_path, w)) for w in model_weights)
    
    if has_weights:
        print(f"  ✓ Model weights found")
    else:
        print(f"  ✗ Model weights not found")
        missing_files.extend(model_weights)
    
    # Final status
    print("\n" + "="*60)
    print("  SETUP COMPLETE")
    print("="*60)
    
    if missing_files:
        print("\n⚠ Warning: Some model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure the BioBERT model files are in:")
        print(f"  {os.path.abspath(model_path)}")
        print("\nYou can download them from the original repository or")
        print("contact the project maintainer.")
    else:
        print("\n✓ All required files are present!")
    
    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60)
    print("\n1. To run the Streamlit app:")
    print("   streamlit run app.py")
    print("\n2. To use Jupyter notebook:")
    print("   jupyter notebook drug_causality_regulatory.ipynb")
    print("\n3. To generate a report programmatically:")
    print("   python")
    print("   >>> from regulatory_causality_report import create_regulatory_report")
    print("   >>> create_regulatory_report('your_file.pdf')")
    print("\n4. For deployment to GitHub:")
    print("   See GITHUB_DEPLOYMENT_GUIDE.md")
    print("\n" + "="*60)
    print("\n✨ Drug Causality BERT v2.0 is ready to use!")
    print("\n")

if __name__ == "__main__":
    main()
