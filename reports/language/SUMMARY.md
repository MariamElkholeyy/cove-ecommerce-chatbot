# Language detection — evaluation of the user-provided Colab model

## What was reused

The original model is a complete scikit-learn Pipeline: character TF-IDF (2–4 characters, 50,000 features, sublinear TF) followed by Logistic Regression (C=1, LBFGS). It was trained in Colab on 70,000 examples. Its saved estimator records 39 solver iterations. No local retraining was performed. The original notebook and model bytes were preserved; the model checksum is recorded in import_provenance.json.

Local scikit-learn was changed from 1.9.0 to 1.6.1 to match the saved model. The remaining original environment versions and original dataset revision are unknown. The local dependency snapshot and evaluation dataset revision are recorded.

## Observed results

| Evaluation | Examples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Original validation split | 10,000 | 99.43% | Not separately reported |
| Original test split | 10,000 | 99.39% | 99.3918% |
| Unique unseen nonconflicting test subset | 9,171 | 99.3894% | 99.3744% |

The locally reproduced validation score matches the original notebook. There are 61 mistakes on the original test split. Test errors and per-class metrics are provided separately.

Exact normalized duplicates were audited. There are 1,029 repeated training text rows, 1,010 repeated validation text rows, and 667 repeated test text rows. Validation contains 132 rows with text seen in training; test contains 188 rows seen in training or validation. These counts overlap and must not simply be added. No training examples were retroactively removed from the existing model. Near duplicates were not detected.

## Uncertainty policy and limitations

Require at least four alphabetic characters and at least one known feature. The confidence threshold was selected using unique validation messages absent from training, with a predeclared target of 99% accepted accuracy and at least 50% coverage. The selected threshold was 0.0: the target was already met. Therefore low confidence alone does not currently reject messages. All original test examples passed the input guards, giving 100% test coverage and the same accepted accuracy as raw accuracy.

Blank input, `12345`, and `OK` now return `unknown`. `unknown` is an abstention, not an identified language. The mixed-language example `عايزة track my order` was accepted as English. The Arabizi example `ana 3ayza el order` was accepted as Turkish. These failures are retained in qualitative_examples.csv. The model is a closed-set classifier and cannot guarantee rejection of unsupported languages. A stricter threshold should be evaluated on a separately reviewed customer-support validation set, with independent testing; it should not be tuned from the existing test errors.

The original notebook trained one approach, so there is no measured baseline comparison. Dataset accuracy does not establish equal performance on dialects, short customer messages, Arabizi, or mixed-language input.

## How to demonstrate

Open notebooks/01_language_detection.ipynb with the project virtual environment. Its saved outputs show the audit, metrics, confusion matrix, and examples. The prediction cell near the end runs independently without training. The original training code and results are preserved in notebooks/reference/Language_Detection_colab_original.ipynb.

From the project terminal: `python -m src.language_detection "Where is my order?"` after activating .venv. The returned object contains language, language_name, candidate, confidence, and reason. Model files are local and excluded from Git; retain them when transferring the project.
