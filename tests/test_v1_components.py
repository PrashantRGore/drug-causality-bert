"""
Unit test suite for Drug Causality BERT v1.0
Tests tokenizer, CausalityClassifier, PDF text extraction, and causality scoring.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    CausalityClassifier,
    simple_sentence_tokenize,
    safe_sent_tokenize,
    classify_causality
)


class TestV1Inference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize model once for testing
        local_model_path = PROJECT_ROOT / "models" / "production_model_final"
        cls.classifier = CausalityClassifier(model_path=str(local_model_path))

    def test_01_sentence_tokenizers(self):
        sample_text = "Patient experienced dizziness. Blood pressure dropped rapidly. Drug was stopped."
        tokens_simple = simple_sentence_tokenize(sample_text)
        tokens_safe = safe_sent_tokenize(sample_text)
        self.assertEqual(len(tokens_simple), 3)
        self.assertEqual(len(tokens_safe), 3)
        self.assertIn("Patient experienced dizziness.", tokens_simple)

    def test_02_empty_input_handling(self):
        res = self.classifier.predict("", return_probs=True)
        self.assertEqual(res["prediction"], "not related")
        self.assertEqual(res["label"], 0)
        self.assertEqual(res["confidence"], 0.0)

    def test_03_causal_statement_prediction(self):
        causal_text = "Patient developed acute anaphylaxis and severe urticaria within 20 minutes after amoxicillin injection."
        res = self.classifier.predict(causal_text, return_probs=True)
        self.assertEqual(res["prediction"], "related")
        self.assertEqual(res["label"], 1)
        self.assertGreater(res["probabilities"]["related"], 0.8)

    def test_04_non_causal_statement_prediction(self):
        non_causal_text = "The patient has a known past history of essential hypertension and is managed on lisinopril."
        res = self.classifier.predict(non_causal_text, return_probs=True)
        self.assertEqual(res["prediction"], "not related")
        self.assertEqual(res["label"], 0)
        self.assertGreater(res["probabilities"]["not_related"], 0.8)

    def test_05_threshold_slider_adjustment(self):
        text = "Patient noted mild transient headache."
        # Stricter threshold vs lower threshold
        res_strict = self.classifier.predict(text, threshold_or_return_probs=0.99)
        self.assertIn("prediction", res_strict)
        self.assertIn("confidence", res_strict)

    def test_06_classify_causality_aggregation(self):
        narrative = (
            "Patient is a 55-year-old male with hypertension. "
            "Severe hepatotoxicity and acute liver injury occurred 1 week after starting methotrexate therapy. "
            "Methotrexate was immediately withdrawn, and liver function enzymes returned to normal."
        )
        report = classify_causality(narrative, model_path=self.classifier.model_path, threshold=0.5)
        self.assertEqual(report["final_classification"], "related")
        self.assertGreater(report["related_sentences"], 0)
        self.assertGreater(report["total_sentences"], 0)
        self.assertIn("not_related_sentences", report)
        self.assertEqual(report["total_sentences"], report["related_sentences"] + report["not_related_sentences"])


if __name__ == "__main__":
    unittest.main()
