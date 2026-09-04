# Cove: local RAG support chat

Start the app by opening **Start Cove.command**, then visit http://127.0.0.1:8000. Leave the terminal running. Stop with Control-C. If port 8000 is already in use by Cove, use the existing app rather than starting another copy.

The Python 3.12 `.venv-rag` environment is separate from your classifier environment. The pinned Torch/NumPy versions support this Intel Mac. The interface runs locally; answer generation sends your question, recent conversation and retrieved references to Groq. No GPU or Colab is required.

## Set up again

From the project folder:

```sh
python3.12 -m venv .venv-rag
.venv-rag/bin/python -m pip install -r requirements-rag-lock.txt
.venv-rag/bin/python scripts/build_rag_index.py
.venv-rag/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep the imported `models/emotion`, `models/intent`, `models/language` and `reports/intent` partitions with the project. Git ignores model/data files; a source-only clone cannot reconstruct these user-trained artifacts by itself. Downloaded embeddings and the FAISS index are cached locally.

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` privately if there is no local key configured. The Connection dialog accepts a session-only replacement. Never put a key in a notebook, a screenshot, a report or source control. Replace the key shared in chat; keep the replacement local. The app binds to loopback and has no multi-user authentication: do not expose this server publicly.

## Understand the pipeline

1. Recognize standalone greetings without retrieval.
2. Use the language detector. Groq translates non-English messages and resolves follow-up references into an English query.
3. Apply your 27-intent SVM. Stronger intent predictions provide a small ranking boost; they do not hard-filter retrieval.
4. Classify emotion on the original English message or its faithful translation, separately from the search rewrite. Low-confidence predictions use neutral wording. Confident negative messages and complaint routes receive a local attention flag.
5. Encode the query with `all-MiniLM-L6-v2`. FAISS searches normalized vectors using inner product (cosine similarity).
6. Pass five relevant question/response pairs to Groq GPT-OSS 20B. The prompt requires grounded support guidance, honest limitations and source IDs.
7. Review the draft for unsupported claims and capability promises, then validate source IDs and show the answer with expandable references.

The RAG corpus contains only questions from the intent training partition. Validation/test questions are excluded. Each question/response pair is a document; embeddings use the question. Long reference responses are capped in the generation context. Cached embeddings avoid rebuilding at every launch.

## What the app can and cannot do

Bitext is a synthetic support dataset. Its example answers include placeholders and simulated agent actions. Cove provides guidance; it cannot check real orders, issue refunds, change an account or contact a human. Numeric store policies and contact details require official confirmation. The UI labels references as examples.

The original fine-tuned DistilBERT is loaded. Its uncalibrated confidence threshold is 0.9; lower-confidence predictions are displayed as uncertain internally and do not assert a customer emotion. Complaints and confident negative messages receive an attention flag only: no escalation is sent. The newer retraining experiment remains unused. See PDF_ALIGNMENT.md for the dataset deviation and measured limitations.

Translation and unsupported-topic handling use the LLM and can be wrong. The saved intent model is not a trained out-of-scope detector. Structured JSON and valid source IDs do not guarantee factual correctness. Review reference relevance and unsupported-claim behavior manually.

## Evaluation and notebook

Open `notebooks/04_rag_chatbot.ipynb` using the `.venv-rag` kernel. It explains retrieval, runs example searches, and separates retrieval from live Groq evaluation.

```sh
.venv-rag/bin/python scripts/evaluate_rag.py
# This also makes live Groq calls:
.venv-rag/bin/python scripts/evaluate_rag.py --generation
```

Reports are in `reports/rag`. Same-intent retrieval hit rates are a proxy for topic relevance, not answer accuracy or a human judgment that every answer is correct. The generation scenario report contains real outputs for manual review. It is a development check, not an independent benchmark.

## Answer safeguards

Standalone English greetings bypass Groq and retrieval. Translated greetings bypass retrieval. Confident pure positive feedback in English/Arabic uses a local acknowledgment without citations; other feedback may return conversation; requests for guidance still require sources. A separate Groq review rejects unsupported advice and action promises; rejected drafts or missing citations receive an honest fallback. This extra review costs one API call per generated answer and can itself make mistakes. Valid source IDs do not prove entailment. There is no real human handoff or transaction integration.

To test the running HTTP application directly, use `.venv-rag/bin/python scripts/check_running_app.py`. This records 15 scenarios, including an Arabic complaint and a contextual follow-up. The latest Arabic complaint triggered an honest fallback; this is a known model-quality limitation, not an HTTP error.
