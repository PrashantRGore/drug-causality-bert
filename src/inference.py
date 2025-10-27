import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CausalityClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, text, threshold=0.5):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        
        pred = 1 if probs[1] > threshold else 0
        return {
            'prediction': 'related' if pred == 1 else 'not related',
            'confidence': float(probs[pred]),
            'probabilities': {
                'not_related': float(probs[0]), 
                'related': float(probs[1])
            }
        }
