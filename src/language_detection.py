"""Load the saved language classifier without retraining it."""
from pathlib import Path
import json
import unicodedata

import joblib

LANGUAGES = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi',
    'it': 'Italian', 'ja': 'Japanese', 'nl': 'Dutch', 'pl': 'Polish',
    'pt': 'Portuguese', 'ru': 'Russian', 'sw': 'Swahili', 'th': 'Thai',
    'tr': 'Turkish', 'ur': 'Urdu', 'vi': 'Vietnamese', 'zh': 'Chinese',
}


def normalize_text(text):
    """Keep scripts and accents; normalize Unicode, case, and whitespace."""
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())


class LanguageDetector:
    def __init__(self, model_dir=None):
        root = Path(__file__).resolve().parents[1]
        model_dir = Path(model_dir) if model_dir else root / 'models/language'
        # Only load artifacts produced by this project: joblib uses pickle.
        self.pipeline = joblib.load(model_dir / 'pipeline.joblib')
        self.metadata = json.loads((model_dir / 'metadata.json').read_text())

    def predict(self, text):
        if not isinstance(text, str):
            raise TypeError('The message must be a string.')
        if len(text) > 10000:
            raise ValueError('Please use a message of at most 10,000 characters.')
        normalized = normalize_text(text)
        if sum(c.isalpha() for c in normalized) < self.metadata['min_letters']:
            return {'language': 'unknown', 'language_name': 'Uncertain',
                    'candidate': None, 'confidence': None, 'reason': 'too_short_or_no_letters'}
        features = self.pipeline.named_steps['tfidf'].transform([text])
        if features.nnz == 0:
            return {'language': 'unknown', 'language_name': 'Uncertain',
                    'candidate': None, 'confidence': None, 'reason': 'no_known_features'}
        classifier = self.pipeline.named_steps['clf']
        probabilities = classifier.predict_proba(features)[0]
        position = int(probabilities.argmax())
        candidate = str(classifier.classes_[position])
        confidence = float(probabilities[position])
        accepted = confidence >= self.metadata['confidence_threshold']
        return {'language': candidate if accepted else 'unknown',
                'language_name': LANGUAGES[candidate] if accepted else 'Uncertain',
                'candidate': candidate, 'confidence': round(confidence, 4),
                'reason': 'accepted' if accepted else 'low_confidence'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Detect the language of a message.')
    parser.add_argument('message')
    args = parser.parse_args()
    print(json.dumps(LanguageDetector().predict(args.message), ensure_ascii=False, indent=2))
