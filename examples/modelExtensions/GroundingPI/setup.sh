#!/usr/bin/env bash
set -euo pipefail
# Run from the starVLA repository root. UV_BIN can point to an existing uv binary.
ENV_DIR=${ENV_DIR:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla}
GROUNDINGPI_SOURCE=${GROUNDINGPI_SOURCE:-/share/project/robot-rfm/robot-exp/usr/cjy/project/GroundingPI}
UV_BIN=${UV_BIN:-uv}
if [[ -e "$ENV_DIR" ]]; then
    echo "Environment already exists: $ENV_DIR" >&2
    exit 1
fi
export UV_SYSTEM_CERTS=true
"$UV_BIN" venv --python 3.12 --seed "$ENV_DIR"
"$ENV_DIR/bin/python" - "$GROUNDINGPI_SOURCE" "$ENV_DIR-sources/transformers" <<'PY'
import hashlib, json, pathlib, sys, tarfile
root, dest = map(pathlib.Path, sys.argv[1:])
entry = json.loads((root / 'third_party/manifest.json').read_text())['dependencies']['transformers']
archive = root / entry['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry['archive_sha256']
dest.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as stream:
    stream.extractall(dest, filter='data')
(dest.parent / 'transformers_revision.json').write_text(json.dumps(entry, indent=2))
PY
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" -r examples/modelExtensions/GroundingPI/requirements.txt -e "$ENV_DIR-sources/transformers" -e .
PYTORCH3D_NO_EXTENSION=1 "$UV_BIN" pip install --python "$ENV_DIR/bin/python" --no-build-isolation \
    "pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba.zip"
"$UV_BIN" pip freeze --python "$ENV_DIR/bin/python" > "$ENV_DIR/installed.freeze.txt"
