import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import CausalityClassifier

st.title("Drug Causality Classifier")
st.caption("BioBERT Model | F1 Score: 97.59%")

@st.cache_resource
def load_model():
    return CausalityClassifier("models/production_model_final")

classifier = load_model()
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

text = st.text_area("Enter medical text:", height=150, 
                   placeholder="e.g., Patient developed rash after taking aspirin...")

if st.button("Classify Text"):
    if text:
        result = classifier.predict(text, threshold)
        
        col1, col2 = st.columns(2)
        
        with col1:
            classification = result['prediction'].upper()
            color = "green" if result['prediction'] == 'related' else "red"
            st.markdown(f"**Classification:** :{color}[{classification}]")
        
        with col2:
            confidence_pct = result['confidence'] * 100
            st.metric("Confidence", f"{confidence_pct:.1f}%")
        
        st.subheader("Probability Distribution")
        probs = result['probabilities']
        st.progress(probs['related'], text=f"Related: {probs['related']:.2%}")
        
        st.json(result)
    else:
        st.warning("Please enter text to classify")

st.markdown("---")
st.caption("Built with BioBERT for pharmacovigilance")
