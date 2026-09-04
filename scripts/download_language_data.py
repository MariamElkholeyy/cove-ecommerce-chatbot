"""Download the assignment's original CSV splits at a fixed revision."""
from pathlib import Path
import hashlib
import json
import shutil
import ssl
import urllib.request

import certifi

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'aa56583bf2bc52b0565770607d6fc3faebecf9e2'
BASE = 'https://huggingface.co/datasets/papluca/language-identification'


def download():
    destination = ROOT / 'data/raw/language'
    destination.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    manifest = {'dataset': 'papluca/language-identification', 'revision': REVISION, 'files': {}}
    manifest_path = destination / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    for split in ['train', 'valid', 'test']:
        path = destination / f'{split}.csv'
        url = f'{BASE}/resolve/{REVISION}/{path.name}'
        if not path.exists():
            partial = path.with_suffix('.partial')
            print(f'Downloading {split}...', flush=True)
            with urllib.request.urlopen(url, context=context, timeout=120) as response:
                with partial.open('wb') as target:
                    shutil.copyfileobj(response, target)
            partial.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous and previous['files'][split]['sha256'] != digest:
            raise ValueError(f'{path.name} differs from its recorded checksum.')
        manifest['files'][split] = {'url': url, 'sha256': digest, 'bytes': path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Dataset files ready; revision and checksums recorded.', flush=True)


if __name__ == '__main__':
    download()
