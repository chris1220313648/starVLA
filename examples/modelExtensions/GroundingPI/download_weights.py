"""Download revision-pinned complete model packages, recording the resolved revisions."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

for repo, folder in [("GroundingPI/GroundingPI", "GroundingPI"), ("physical-intelligence/fast", "fast")]:
    dest = Path("playground/Pretrained_models") / folder
    record = dest / "download_revision.json"
    revision = json.loads(record.read_text())["revision"] if record.exists() else HfApi().model_info(repo).sha
    snapshot_download(repo, revision=revision, local_dir=str(dest), max_workers=4)
    record.write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
    print(repo, revision, flush=True)
