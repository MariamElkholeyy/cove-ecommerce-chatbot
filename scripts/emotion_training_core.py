"""Shared functions embedded in the self-contained Colab training notebook."""
# SECTION: imports
from pathlib import Path
import copy
import hashlib
import json
import random
import re
import shutil
import ssl
import subprocess
import sys
import time
import unicodedata
import urllib.request
import warnings
from importlib.metadata import version
from datetime import datetime, timezone

import certifi
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

DATASET_ID = 'cardiffnlp/tweet_eval'
DATASET_REVISION = 'b3a375baf0f409c77e6bc7aa35102b7b3534f8be'
MODEL_ID = 'distilbert/distilbert-base-uncased'
MODEL_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
ID2LABEL = {0: 'sad', 1: 'neutral', 2: 'happy'}
LABEL2ID = {name: number for number, name in ID2LABEL.items()}
SOURCE_LABELS = ['negative', 'neutral', 'positive']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# SECTION: data
def clean_emotion_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'(?<!\w)@[\w]+', '@user', text)
    return ' '.join(text.split())


def download_emotion_data(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': DATASET_ID, 'subset': 'sentiment', 'revision': DATASET_REVISION, 'files': {}}
    manifest_path = destination / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    frames = {}
    context = ssl.create_default_context(cafile=certifi.where())
    for split in ['train', 'validation', 'test']:
        path = destination / f'{split}.parquet'
        url = f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/sentiment/{split}-00000-of-00001.parquet'
        if not path.exists():
            temporary = path.with_suffix('.partial')
            with urllib.request.urlopen(url, context=context, timeout=120) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous and previous['files'][split]['sha256'] != digest:
            raise ValueError(f'{split} data differs from its recorded checksum.')
        table = pq.read_table(path)
        schema_info = json.loads(table.schema.metadata[b'huggingface'])
        assert schema_info['info']['features']['label']['names'] == SOURCE_LABELS
        frames[split] = table.to_pandas()
        assert set(frames[split].label) == {0, 1, 2}
        manifest['files'][split] = {'sha256': digest, 'url': url, 'rows': len(frames[split])}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return frames, manifest


def prepare_emotion_splits(raw):
    cleaned, audit, seen = {}, [], set()
    for split in ['train', 'validation', 'test']:
        frame = raw[split].dropna(subset=['text', 'label']).copy()
        frame['text'] = frame.text.map(clean_emotion_text)
        frame = frame[frame.text.str.len() > 0].copy()
        frame['key'] = frame.text.str.casefold()
        conflicts = set(frame.groupby('key').label.nunique().loc[lambda x: x > 1].index)
        conflict_rows = int(frame.key.isin(conflicts).sum())
        frame = frame[~frame.key.isin(conflicts)]
        duplicate_rows = int(frame.duplicated('key').sum())
        frame = frame.drop_duplicates('key')
        overlap_rows = int(frame.key.isin(seen).sum())
        frame = frame[~frame.key.isin(seen)].copy()
        seen.update(frame.key)
        frame = frame[['text', 'label']].reset_index(drop=True)
        assert set(frame.label) == {0, 1, 2}
        cleaned[split] = frame
        audit.append({'split': split, 'original': len(raw[split]), 'retained': len(frame),
                      'conflicting_rows': conflict_rows, 'within_split_duplicate_rows': duplicate_rows,
                      'earlier_split_overlap_rows': overlap_rows})
    return cleaned, pd.DataFrame(audit)


# SECTION: batches
class EmotionDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame.text.tolist()
        self.labels = frame.label.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {'text': self.texts[index], 'label': self.labels[index]}


class EmotionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, examples):
        batch = self.tokenizer([example['text'] for example in examples], padding=True,
                               truncation=True, max_length=self.max_length, return_tensors='pt')
        batch['labels'] = torch.tensor([example['label'] for example in examples], dtype=torch.long)
        return batch


def make_loader(frame, tokenizer, batch_size, max_length, shuffle=False, seed=42):
    return DataLoader(EmotionDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=EmotionCollator(tokenizer, max_length), num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict_text_probabilities(model, tokenizer, texts, device, batch_size=32, max_length=128):
    model.eval()
    probabilities = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer([clean_emotion_text(t) for t in texts[start:start + batch_size]],
                            padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {key: tensor.to(device) for key, tensor in encoded.items()}
        probabilities.extend(model(**inputs).logits.softmax(dim=-1).cpu().tolist())
    return np.asarray(probabilities)


@torch.inference_mode()
def evaluate_emotion(model, loader, device):
    model.eval()
    truths, probabilities = [], []
    for batch in loader:
        labels = batch.pop('labels')
        inputs = {key: tensor.to(device) for key, tensor in batch.items()}
        logits = model(**inputs).logits
        truths.extend(labels.tolist())
        probabilities.extend(logits.softmax(dim=-1).cpu().tolist())
    truth, probabilities = np.asarray(truths), np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(truth, predicted)),
            'macro_f1': float(f1_score(truth, predicted, average='macro', labels=[0,1,2], zero_division=0))}, truth, probabilities


# SECTION: training
def train_emotion(model, tokenizer, splits, output_dir, device, epochs=3, batch_size=32,
                  max_length=128, learning_rate=2e-5, seed=42):
    """Select the checkpoint with best validation macro-F1; never use test data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / 'best_model'
    train_loader = make_loader(splits['train'], tokenizer, batch_size, max_length, True, seed)
    val_loader = make_loader(splits['validation'], tokenizer, batch_size, max_length)
    counts = splits['train'].label.value_counts().reindex([0,1,2]).to_numpy()
    weights = torch.tensor(len(splits['train']) / (3 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    model.to(device)
    history, best_f1, best_epoch = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, batches, started = 0.0, 0, time.perf_counter()
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch.pop('labels').to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss: stop and investigate.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            loss_sum += loss.item()
            batches += 1
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(f'Epoch {epoch}/{epochs}, batch {batch_index}/{len(train_loader)}, mean weighted loss {loss_sum/batches:.4f}', flush=True)
        validation, _, _ = evaluate_emotion(model, val_loader, device)
        record = {'epoch': epoch, 'mean_weighted_training_loss': loss_sum / batches,
                  'seconds': time.perf_counter() - started, **validation}
        history.append(record)
        print('Validation:', record, flush=True)
        if validation['macro_f1'] > best_f1:
            best_f1, best_epoch = validation['macro_f1'], epoch
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        (output_dir / 'training_history.json').write_text(json.dumps(history, indent=2) + '\n')
    return best_dir, history, {'best_epoch': best_epoch, 'best_validation_macro_f1': best_f1,
                               'class_weights': weights.cpu().tolist()}


# SECTION: review
def choose_review_threshold(truth, probabilities, target_accuracy=0.85, min_coverage=0.30):
    """A review flag does not change the three predicted class labels."""
    predicted, scores = probabilities.argmax(axis=1), probabilities.max(axis=1)
    rows = []
    for threshold in [0.0,0.4,0.5,0.6,0.7,0.8,0.9,0.95]:
        accepted = scores >= threshold
        rows.append({'threshold': threshold, 'coverage': float(accepted.mean()),
                     'accepted_accuracy': float((predicted[accepted] == truth[accepted]).mean()) if accepted.any() else None})
    table = pd.DataFrame(rows)
    acceptable = table[(table.coverage >= min_coverage) & (table.accepted_accuracy >= target_accuracy)]
    threshold = float(acceptable.iloc[0].threshold) if len(acceptable) else 1.0
    return threshold, table


def package_emotion_export(model_dir, report_dir, metadata, archive_base):
    """Bundle weights, tokenizer, configuration, metadata, reports, and checksums."""
    staging = Path(archive_base).parent / 'export_bundle'
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(model_dir, staging / 'model')
    (staging / 'model/metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copytree(report_dir, staging / 'reports')
    manifest = {}
    for path in sorted(staging.rglob('*')):
        if path.is_file():
            manifest[path.relative_to(staging).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return Path(shutil.make_archive(str(archive_base), 'zip', staging))
