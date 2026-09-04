#!/bin/zsh
cd -- "${0:A:h}"
if [[ ! -x .venv-rag/bin/python ]]; then
  echo 'Run the RAG setup steps in RAG_GUIDE.md first.'
  read '?Press Enter to close.'
  exit 1
fi
.venv-rag/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
