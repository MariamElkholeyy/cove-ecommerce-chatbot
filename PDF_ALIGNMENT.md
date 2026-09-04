# Assignment alignment and assessment notes

Reviewed against Customer_Support_Chatbot_Task.pdf on 2026-09-04.

## What is implemented

| Requirement | Implementation and evidence |
|---|---|
| Traditional language classification | Character TF-IDF + Logistic Regression; papluca language identification; notebook 01 and reports/language. |
| Transformer/RNN emotion classification | Fine-tuned DistilBERT; happy/neutral/sad are positive/neutral/negative. Executed notebook 02 and models/emotion/metadata.json. |
| Classical or zero/few-shot intent classification | Word TF-IDF + LinearSVC; 27 Bitext labels mapped to broader routes. Executed notebook 03 and reports/intent. |
| Vector retrieval | MiniLM, normalized 384-dimensional embeddings, local FAISS IndexFlatIP. |
| Support knowledge base | 19,430 Bitext training question/response pairs. Questions embedded; paired responses supplied as context. Validation/test queries excluded. |
| Groq LLM | GPT-OSS 20B through the server-side API key. No LLM fine-tuning. |
| Complaint/negative route | Complaint routes and confident negative sentiment get a local attention flag. Negative tone requests empathy. No actual escalation occurs. |
| Same-language response | Local language detector, English translation for English-only models, requested-language generation. Needs multilingual human review. |
| Grounded guidance | Retrieved references, source ID validation, separate draft review, honest fallback. These reduce errors but are not a proof of correctness. |
| Python local deployment | FastAPI, local models and browser interface; Start Cove.command. |
| Four notebooks and deployment scripts | notebooks/01 through 04, scripts/, app/. The imported Colab outputs are preserved, not regenerated. |

## Dataset deviation to disclose

The PDF names dair-ai/emotion, whose listed six labels have no genuine neutral class. This project uses TweetEval sentiment, which provides positive/neutral/negative labels, and maps those to the TA's happy/neutral/sad names. This supports the requested semantics but differs from the named dataset. Only the TA can confirm acceptance of that substitution. Do not state that dair-ai/emotion was used or that the TA approved the substitute.

The PDF does not set a numerical accuracy minimum. The separate verbal request for emotion accuracy in the 90s is not achieved: the deployed model has 68.42% test accuracy and 68.37% macro-F1. Integration fixes do not improve this measured benchmark. A separate training experiment would be necessary to investigate improved performance, with no guarantee of 90%.

## Training versus running

Language and intent use trained classical models locally. DistilBERT was fine-tuned on Colab and is now loaded locally on CPU. MiniLM is pretrained and only computes embeddings. FAISS stores/searches vectors; it is not a generative model. Groq supplies hosted generation and translation. No Colab session, GPU, dataset download or retraining is required for current local testing.

## Routing decisions and why

Standalone English greetings/thanks/goodbyes use local answers. Common explicit human requests use a fixed honest response: contact the store yourself; no transfer is available. Other languages and phrasing use the generation path. Translated standalone social intents skip retrieval. Feedback without a factual question uses conversation status and does not require citations or claim submission of a review.

For support questions, the SVM intent gives a small ranking boost when its margin is strong. We do not hard-filter the corpus, because a mistaken intent could hide the right references. Complaints/negative sentiment alter priority and tone, while the answer can still retrieve guidance. Unsupported topics and ambiguous requests are handled by the generator, not a trained out-of-scope classifier.

Emotion is computed from original English wording or a separate faithful translation, never a follow-up search rewrite. Predictions below the saved 0.9 threshold are uncertain and use warm neutral wording. Even a high-confidence prediction can be wrong: the scores are not calibrated. Priority means attention suggested, not faster service or human notification.

## Honest boundaries

Bitext is synthetic; it is not verified store policy or live records. The application cannot issue refunds, view orders, edit accounts, send a complaint, or contact a person. Do not demonstrate it as if those actions occurred. Generated references may contain fictional first-person statements; the answer prompt and separate review reject adopting them. Draft review is itself an LLM judgment and can miss unsupported statements or reject a useful answer. A fallback is preferable to an invented policy, but still counts as a quality limitation.

## Verification

Run `.venv-rag/bin/python -m unittest discover -s tests -v` for offline regression checks. Run `.venv-rag/bin/python scripts/evaluate_rag.py --generation` for retrieval plus live behavior checks; it uses the configured Groq key and is paced for free-tier limits. Reports record actual responses and API failures. Review them manually; do not count successful HTTP requests as correct answers.

Retrieval's same-intent hit@5 of 100% on 135 sampled test questions is a topic proxy, not chatbot accuracy. Language accuracy is 99.39%; intent test accuracy is 99.71% on the saved benchmark. Near-paraphrases and domain shift limit generalization. Keep training/validation/test partitions separate and do not repeatedly tune on test results.

## Observed integration check results — 2026-09-04

Nine offline regression tests passed. Fifteen live HTTP scenarios completed without HTTP errors in the final combined report. Happy feedback and the Arabic complaint were retested after their targeted fixes. These are development observations, not an independent accuracy score. The Arabic complaint received an honest fallback because the draft proposed cancellation instead of addressing a delay; the safeguard detected and blocked that mismatch. Review reports/rag/generation_checks.json and integration_verification.json for evidence.

The final RAG notebook reads these real saved HTTP results and executes local retrieval checks without further API charges. Its commented examples explain fresh generation. The imported language/emotion/intent notebooks retain original executed outputs.
