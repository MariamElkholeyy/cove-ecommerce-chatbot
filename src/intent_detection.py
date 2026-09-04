"""Load a trusted classical intent export and predict English customer intents."""
from pathlib import Path
import json
import re
import joblib
import numpy as np

class IntentDetector:
    def __init__(self, model_dir=None):
        directory = Path(model_dir) if model_dir else Path(__file__).resolve().parent / 'model'
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        self.pipeline = joblib.load(directory / 'pipeline.joblib')
        self.routes = self.metadata['routes']
        if set(self.pipeline.classes_) != set(self.routes):
            raise ValueError('Pipeline labels do not match routing metadata.')

    def predict(self, text, language='en'):
        if language != 'en':
            raise ValueError('Translate the message into English first.')
        if not isinstance(text,str) or not text.strip() or len(text)>10000:
            raise ValueError('Use a nonempty string of at most 10,000 characters.')
        social = re.sub(r'[^\w\s]', '', text.casefold()).strip()
        social = ' '.join(social.split())
        for label, phrases in {
            'greeting': {'hi','hello','hey','good morning','good evening'},
            'goodbye': {'bye','goodbye','see you','bye for now'},
            'gratitude': {'thanks','thank you','thanks a lot','thank you very much'},
        }.items():
            if social in phrases:
                return {'intent':label,'candidate_intent':None,'route':label,'margin':None,
                        'needs_review':False,'priority':False,'reason':'standalone_social_rule'}
        matrix = self.pipeline.named_steps['tfidf'].transform([text])
        if matrix.nnz == 0:
            return {'intent':'needs_clarification','candidate_intent':None,
                    'route':'needs_clarification','margin':None,'needs_review':True,
                    'priority':False,'reason':'no_known_features'}
        clf = self.pipeline.named_steps['classifier']
        scores = clf.decision_function(matrix)[0]
        candidate = str(clf.classes_[scores.argmax()])
        ordered = np.sort(scores)
        margin = float(ordered[-1]-ordered[-2])
        review = self.metadata['review_all'] or margin < self.metadata['margin_threshold']
        label = 'needs_clarification' if review else candidate
        return {'intent':label,'candidate_intent':candidate,
                'route':'needs_clarification' if review else self.routes[candidate],
                'margin':margin,'needs_review':bool(review),
                'priority':self.routes[candidate] in {'complaint','human_handoff'},
                'reason':'uncertain_prediction' if review else 'model_prediction'}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    print(json.dumps(IntentDetector().predict(args.message),indent=2))
