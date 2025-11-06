# Root-level wrapper for Streamlit Cloud deployment
# This imports and runs the actual app from app/streamlit_app.py

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the actual app
from app.streamlit_app import *
