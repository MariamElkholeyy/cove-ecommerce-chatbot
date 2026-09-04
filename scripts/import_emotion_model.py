"""Verify and import a completed Colab export; never silently overwrite a model."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def import_export(archive, root=ROOT):
    root = Path(root)
    target = root / 'models/emotion'
    if target.exists():
        raise FileExistsError('An emotion model is already installed. Back it up before importing a replacement.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries.')
        if sum(entry.file_size for entry in bundle.infolist()) > 2_000_000_000:
            raise ValueError('Export exceeds the expected size limit.')
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
                raise ValueError('Unsafe path in ZIP export.')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Symbolic links are not supported in model exports.')
        manifest = json.loads(bundle.read('manifest.json'))
        files = {entry.filename for entry in bundle.infolist() if not entry.is_dir()}
        if files != set(manifest) | {'manifest.json'}:
            raise ValueError('Export files do not match the checksum manifest.')
        for name, digest in manifest.items():
            if not name.startswith(('model/', 'reports/')):
                raise ValueError('Unexpected export directory.')
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Checksum mismatch: {name}')
        metadata = json.loads(bundle.read('model/metadata.json'))
        config = json.loads(bundle.read('model/config.json'))
        if metadata.get('training_status') != 'full_training_completed':
            raise ValueError('This is not a completed full training run. Smoke tests are not deployable models.')
        expected = {'0': 'sad', '1': 'neutral', '2': 'happy'}
        if config.get('id2label') != expected or metadata.get('id2label') != expected:
            raise ValueError('Unexpected three-class label mapping.')
        for required in ['model/model.safetensors', 'model/tokenizer_config.json', 'model/tokenizer.json']:
            if required not in files:
                raise ValueError(f'Missing required model file: {required}')
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            bundle.extractall(temporary)
            staged = Path(temporary)
            report_target = root / 'reports/emotion' / datetime.now(timezone.utc).strftime('colab-%Y%m%d-%H%M%S-%f')
            shutil.copytree(staged / 'reports', report_target)
            shutil.copytree(staged / 'model', target)
            (report_target / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Imported model to {target}')
    print(f'Evaluation reports: {report_target}')
    print('Install compatible inference dependencies before using EmotionDetector.')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    import_export(args.archive)
