"""Three-class English sentiment inference: sad, neutral, happy."""
from pathlib import Path
import json
import re
import unicodedata


def clean_emotion_text(text):
    """Preserve sentiment clues such as punctuation, emoji, and negation."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'(?<!\w)@[\w]+', '@user', text)
    return ' '.join(text.split())


class EmotionDetector:
    def __init__(self, model_dir=None, device='cpu'):
        root = Path(__file__).resolve().parents[1]
        directory = Path(model_dir) if model_dir else root / 'models/emotion'
        if not (directory / 'metadata.json').exists():
            raise FileNotFoundError('Train the Colab notebook and import its export first. No trained emotion model is installed yet.')
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata.get('training_status') != 'full_training_completed':
            raise ValueError('This artifact is not a completed full training run.')
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True).to(self.device)
        self.model.eval()
        if self.model.config.id2label != {0: 'sad', 1: 'neutral', 2: 'happy'}:
            raise ValueError('Unexpected label mapping in the model configuration.')

    def predict(self, text, language='en'):
        if language != 'en':
            raise ValueError('This classifier expects English. Translate non-English messages before calling it.')
        if not isinstance(text, str):
            raise TypeError('The message must be a string.')
        if len(text) > 10000:
            raise ValueError('Use a message of at most 10,000 characters.')
        cleaned = clean_emotion_text(text)
        if not cleaned:
            raise ValueError('The message must contain text; empty input is not neutral sentiment.')
        encoded = self.tokenizer(cleaned, return_tensors='pt', truncation=True,
                                 max_length=self.metadata['max_length'])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            probabilities = self.model(**encoded).logits.softmax(dim=-1)[0].cpu().tolist()
        predicted = max(range(3), key=lambda i: probabilities[i])
        return {'label': self.model.config.id2label[predicted],
                'confidence': round(probabilities[predicted], 4),
                'scores': {self.model.config.id2label[i]: round(p, 4) for i, p in enumerate(probabilities)},
                'needs_review': self.metadata['review_confidence_threshold'] >= 1.0 or probabilities[predicted] < self.metadata['review_confidence_threshold']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classify English customer sentiment.')
    parser.add_argument('message')
    args = parser.parse_args()
    print(json.dumps(EmotionDetector().predict(args.message), indent=2))
