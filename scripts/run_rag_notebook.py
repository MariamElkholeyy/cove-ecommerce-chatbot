"""Execute the educational RAG notebook with this Python interpreter."""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/04_rag_chatbot.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
try:
    NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    print('RAG notebook executed successfully with real retrieval and Groq outputs.')
finally:
    if km.has_kernel:km.shutdown_kernel(now=True)
