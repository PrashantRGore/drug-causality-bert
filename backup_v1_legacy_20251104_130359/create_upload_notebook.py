import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Drug Causality Classification - Enhanced with File Upload\n",
                "\n",
                "Upload PDF files directly and analyze drug causality using BioBERT.\n",
                "\n",
                "## Performance Metrics\n",
                "- F1 Score: 0.9759\n",
                "- Accuracy: 0.9759\n",
                "- Sensitivity: 0.9868\n",
                "- Specificity: 0.9650"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Install Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "!pip install torch transformers pandas numpy scikit-learn nltk PyPDF2 safetensors ipywidgets -q\n",
                "print('✓ Packages installed!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Import Libraries and Load Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "sys.path.insert(0, str(Path.cwd() / 'src'))\n",
                "\n",
                "import torch\n",
                "from inference import CausalityClassifier, classify_causality, extract_text_from_pdf\n",
                "import json\n",
                "from datetime import datetime\n",
                "import os\n",
                "\n",
                "print('✓ Libraries imported successfully!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the model\n",
                "MODEL_PATH = './models/production_model_final'\n",
                "THRESHOLD = 0.5\n",
                "\n",
                "print('Loading model...')\n",
                "classifier = CausalityClassifier(model_path=MODEL_PATH, threshold=THRESHOLD)\n",
                "print('✓ Model loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Upload and Analyze PDF File\n",
                "\n",
                "Use the file upload widget below to select and analyze your PDF document."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ipywidgets import FileUpload, Button, VBox, HTML, Output\n",
                "from IPython.display import display, clear_output\n",
                "import io\n",
                "\n",
                "# Create upload widget\n",
                "upload_widget = FileUpload(accept='.pdf', multiple=False)\n",
                "analyze_button = Button(description='Analyze PDF', button_style='success', disabled=True)\n",
                "output_area = Output()\n",
                "\n",
                "uploaded_file_path = None\n",
                "\n",
                "def save_uploaded_file(change):\n",
                "    global uploaded_file_path\n",
                "    if upload_widget.value:\n",
                "        uploaded_file = list(upload_widget.value.values())[0]\n",
                "        filename = uploaded_file['metadata']['name']\n",
                "        content = uploaded_file['content']\n",
                "        \n",
                "        # Save to data/raw directory\n",
                "        os.makedirs('./data/raw', exist_ok=True)\n",
                "        uploaded_file_path = f'./data/raw/{filename}'\n",
                "        \n",
                "        with open(uploaded_file_path, 'wb') as f:\n",
                "            f.write(content)\n",
                "        \n",
                "        analyze_button.disabled = False\n",
                "        \n",
                "        with output_area:\n",
                "            clear_output()\n",
                "            print(f'✓ File uploaded: {filename}')\n",
                "            print(f'✓ Saved to: {uploaded_file_path}')\n",
                "            print('\\nClick \"Analyze PDF\" button to process.')\n",
                "\n",
                "def analyze_pdf(b):\n",
                "    global uploaded_file_path\n",
                "    with output_area:\n",
                "        if not uploaded_file_path or not os.path.exists(uploaded_file_path):\n",
                "            print('⚠ Please upload a PDF file first!')\n",
                "            return\n",
                "        \n",
                "        clear_output()\n",
                "        print(f'Analyzing: {uploaded_file_path}\\n')\n",
                "        print('=' * 80)\n",
                "        \n",
                "        try:\n",
                "            # Extract text\n",
                "            pdf_text = extract_text_from_pdf(uploaded_file_path)\n",
                "            print(f'✓ Extracted {len(pdf_text)} characters from PDF\\n')\n",
                "            \n",
                "            # Classify\n",
                "            results = classify_causality(pdf_text, MODEL_PATH, THRESHOLD, verbose=True)\n",
                "            \n",
                "            # Display results\n",
                "            print('\\n' + '=' * 80)\n",
                "            print('RESULTS')\n",
                "            print('=' * 80)\n",
                "            print(f\"Final Classification: {results['final_classification'].upper()}\")\n",
                "            print(f\"Confidence Score: {results['confidence_score']:.4f}\")\n",
                "            print(f\"Related Sentences: {results['related_sentences']} / {results['total_sentences']}\")\n",
                "            \n",
                "            if results['top_related_sentences']:\n",
                "                print('\\nTop 5 Related Sentences:')\n",
                "                print('-' * 80)\n",
                "                for i, sent in enumerate(results['top_related_sentences'], 1):\n",
                "                    print(f\"\\n{i}. {sent['sentence']}\")\n",
                "                    print(f\"   Probability: {sent['probability_related']:.4f}\")\n",
                "            \n",
                "            # Save results\n",
                "            os.makedirs('./results', exist_ok=True)\n",
                "            filename = Path(uploaded_file_path).stem\n",
                "            result_path = f'./results/{filename}_report.json'\n",
                "            with open(result_path, 'w') as f:\n",
                "                json.dump(results, f, indent=2)\n",
                "            print(f'\\n✓ Results saved to: {result_path}')\n",
                "            \n",
                "        except Exception as e:\n",
                "            print(f'❌ Error: {str(e)}')\n",
                "            import traceback\n",
                "            traceback.print_exc()\n",
                "\n",
                "upload_widget.observe(save_uploaded_file, names='value')\n",
                "analyze_button.on_click(analyze_pdf)\n",
                "\n",
                "display(VBox([\n",
                "    HTML('<h3>📄 Upload PDF File</h3>'),\n",
                "    upload_widget,\n",
                "    analyze_button,\n",
                "    output_area\n",
                "]))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Test with Custom Text\n",
                "\n",
                "You can also test the model with custom text without uploading a PDF."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Enter your custom text here\n",
                "your_text = \"\"\"The patient developed severe rash after taking the medication.\"\"\"\n",
                "\n",
                "if your_text:\n",
                "    result = classifier.predict(your_text, return_probs=True)\n",
                "    print('=' * 80)\n",
                "    print(f'Text: {your_text}')\n",
                "    print(f\"\\nPrediction: {result['prediction'].upper()}\")\n",
                "    print(f\"Confidence: {result['confidence']:.4f}\")\n",
                "    print(f\"\\nProbabilities:\")\n",
                "    print(f\"  - Not Related: {result['probabilities']['not_related']:.4f}\")\n",
                "    print(f\"  - Related: {result['probabilities']['related']:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Batch Process Multiple PDFs\n",
                "\n",
                "Process all PDF files in the data/raw directory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from inference import process_multiple_pdfs\n",
                "import glob\n",
                "\n",
                "# Find all PDF files\n",
                "pdf_files = glob.glob('./data/raw/*.pdf')\n",
                "\n",
                "if pdf_files:\n",
                "    print(f'Found {len(pdf_files)} PDF files\\n')\n",
                "    \n",
                "    # Process all PDFs\n",
                "    batch_results = process_multiple_pdfs(\n",
                "        pdf_paths=pdf_files,\n",
                "        model_path=MODEL_PATH,\n",
                "        threshold=THRESHOLD,\n",
                "        save_reports=True,\n",
                "        output_dir='./results'\n",
                "    )\n",
                "    \n",
                "    # Display summary\n",
                "    print('=' * 80)\n",
                "    print('BATCH PROCESSING SUMMARY')\n",
                "    print('=' * 80)\n",
                "    \n",
                "    for i, result in enumerate(batch_results, 1):\n",
                "        print(f\"\\n{i}. {result['pdf_file']}\")\n",
                "        if 'error' in result:\n",
                "            print(f\"   Status: ERROR - {result['error']}\")\n",
                "        else:\n",
                "            print(f\"   Classification: {result['final_classification'].upper()}\")\n",
                "            print(f\"   Confidence: {result['confidence_score']:.4f}\")\n",
                "            print(f\"   Related Sentences: {result['related_sentences']}/{result['total_sentences']}\")\n",
                "    \n",
                "    # Save batch summary\n",
                "    summary_path = './results/batch_causality_summary.json'\n",
                "    with open(summary_path, 'w') as f:\n",
                "        json.dump(batch_results, f, indent=2)\n",
                "    \n",
                "    print(f'\\n✓ Batch summary saved to {summary_path}')\n",
                "else:\n",
                "    print('No PDF files found in ./data/raw/ directory')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook
with open('drug_causality_upload.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✓ Notebook created: drug_causality_upload.ipynb")
