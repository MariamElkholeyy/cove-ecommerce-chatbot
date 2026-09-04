"""Restore a trusted runtime bundle after validating its manifest and paths."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, stat, tempfile, zipfile
ROOT=Path(__file__).resolve().parents[1]
NAMES={'language','intent','emotion','embedding','rag'}

def restore(archive):
    expected=json.loads((ROOT/'docs/artifact_checksums.json').read_text())
    if any((ROOT/'models'/name).exists() for name in NAMES):
        raise FileExistsError('Model directories already exist. Back them up before restoring another bundle.')
    with zipfile.ZipFile(archive) as z:
        files=z.infolist()
        if len(files)!=len({f.filename for f in files}):raise ValueError('Duplicate archive entries.')
        if sum(f.file_size for f in files)>2_000_000_000:raise ValueError('Archive exceeds the expected size limit.')
        if set(z.namelist())!=set(expected)|{'ARTIFACT_CHECKSUMS.json'}:raise ValueError('Unexpected archive contents.')
        if json.loads(z.read('ARTIFACT_CHECKSUMS.json'))!=expected:raise ValueError('Manifest does not match this source version.')
        for f in files:
            p=PurePosixPath(f.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in f.filename or stat.S_ISLNK(f.external_attr>>16):
                raise ValueError('Unsafe archive entry.')
            if f.filename=='ARTIFACT_CHECKSUMS.json':continue
            if len(p.parts)<3 or p.parts[0]!='models' or p.parts[1] not in NAMES:raise ValueError('Unexpected model path.')
            with z.open(f) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected[f.filename]:raise ValueError('Artifact checksum mismatch.')
        with tempfile.TemporaryDirectory(prefix='artifact-restore-',dir=ROOT) as tmp:
            z.extractall(tmp)
            (ROOT/'models').mkdir(exist_ok=True)
            for name in sorted(NAMES):shutil.move(str(Path(tmp)/'models'/name),str(ROOT/'models'/name))
    print('Restored all five model directories. Configure .env and start the app.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);restore(p.parse_args().archive)
