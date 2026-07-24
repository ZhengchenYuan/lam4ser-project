#!/bin/bash
# Create an isolated environment for Qwen3-Omni teacher inference.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUDIOQWEN3_ENV="${AUDIOQWEN3_ENV:-/data/chi-gpu4/ge94xov/audioqwen3-env}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CLUSTER_LIBSTDCXX=/data/chi-gpu4/eihwadmin/.cache/rattler/cache/pkgs/libstdcxx-15.1.0-h8f9b012_5/lib

# Match the runtime environment used by the pilot Slurm job. This is required
# on chi-gpu1 for PyTorch/NumPy shared-library loading.
export LD_LIBRARY_PATH="$CLUSTER_LIBSTDCXX:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"

if [[ "$AUDIOQWEN3_ENV" == "/data/chi-gpu4/ge94xov/lam4ser-env" ]]; then
  echo "Refusing to modify lam4ser-env; choose a dedicated AUDIOQWEN3_ENV." >&2
  exit 2
fi

"$BOOTSTRAP_PYTHON" -m venv "$AUDIOQWEN3_ENV"
PYTHON="$AUDIOQWEN3_ENV/bin/python"

"$PYTHON" -m pip install --upgrade pip "setuptools<82" wheel
"$PYTHON" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install \
  "transformers>=5.2.0" \
  accelerate \
  qwen-omni-utils \
  soundfile \
  imageio-ffmpeg

# qwen-omni-utils expects an ffmpeg executable. Always expose imageio-ffmpeg's
# bundled binary inside this venv so the result does not depend on login-node
# modules or the submitting shell's PATH.
FFMPEG_BINARY="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG_BINARY" "$AUDIOQWEN3_ENV/bin/ffmpeg"
export PATH="$AUDIOQWEN3_ENV/bin:$PATH"

echo
echo "Created isolated AudioQwen3 environment: $AUDIOQWEN3_ENV"
echo "lam4ser-env was not modified."
echo
"$PYTHON" scripts/generate_audioqwen3_teacher_rationales.py --check_dependencies
