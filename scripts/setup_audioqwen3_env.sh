#!/bin/bash
# Create an isolated environment for Qwen3-Omni teacher inference.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUDIOQWEN3_ENV="${AUDIOQWEN3_ENV:-/data/chi-gpu4/ge94xov/audioqwen3-env}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ "$AUDIOQWEN3_ENV" == "/data/chi-gpu4/ge94xov/lam4ser-env" ]]; then
  echo "Refusing to modify lam4ser-env; choose a dedicated AUDIOQWEN3_ENV." >&2
  exit 2
fi

"$BOOTSTRAP_PYTHON" -m venv "$AUDIOQWEN3_ENV"
PYTHON="$AUDIOQWEN3_ENV/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install \
  "transformers>=5.2.0" \
  accelerate \
  qwen-omni-utils \
  soundfile \
  imageio-ffmpeg

# qwen-omni-utils expects an ffmpeg executable. Prefer the cluster binary; if
# it is absent, expose imageio-ffmpeg's bundled binary inside this venv only.
if ! command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BINARY="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  ln -sf "$FFMPEG_BINARY" "$AUDIOQWEN3_ENV/bin/ffmpeg"
fi

echo
echo "Created isolated AudioQwen3 environment: $AUDIOQWEN3_ENV"
echo "lam4ser-env was not modified."
echo
"$PYTHON" scripts/generate_audioqwen3_teacher_rationales.py --check_dependencies
