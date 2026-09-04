# Runtime artifacts

The complete app requires five model directories: language, intent, emotion, embedding and rag. They are packaged separately as Cove_Runtime_Artifacts.zip, not committed to Git.

The archive contains only the model directories and a checksum manifest. It does not contain the API key, virtual environments, raw downloads, browser state or Git history. Request it from the project owner until a release download is published. The repository records the expected checksums in artifact_checksums.json.

Run `python scripts/restore_artifacts.py /path/to/Cove_Runtime_Artifacts.zip` from a configured environment. Existing models are never overwritten automatically. The archive represents the deployed checkpoints, not a new training run.

Classical joblib artifacts are executable pickle formats: use a trusted archive. Keep the checksum file with the source version. Third-party model and dataset terms still apply.
