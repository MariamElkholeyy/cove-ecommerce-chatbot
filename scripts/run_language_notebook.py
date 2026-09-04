"""Execute the evaluation notebook with this project's Python environment."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
for key, directory in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'),
                       ('JUPYTER_RUNTIME_DIR', 'jupyter')]:
    path = ROOT / '.cache' / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'

import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

install(prefix=str(ROOT / '.venv'), kernel_name='iti-chatbot',
        display_name='Python (ITI ChatBot .venv)')
path = ROOT / 'notebooks/01_language_detection.ipynb'
notebook = nbformat.read(path, as_version=4)


def progress(cell, cell_index, **kwargs):
    print(f'Cell {cell_index + 1}: {cell.source.splitlines()[0][:100]}', flush=True)


try:
    NotebookClient(notebook, timeout=1200, kernel_name='iti-chatbot',
                   resources={'metadata': {'path': str(ROOT)}},
                   on_cell_start=progress).execute()
finally:
    nbformat.write(notebook, path)
print('Executed notebook saved.', flush=True)
