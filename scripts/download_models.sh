#!/usr/bin/env bash
#
# MemoryNav — scripts/download_models.sh
#
# Pulls pretrained model weights into data/models/ so Module 1
# (Perception Layer) has what it needs. Safe to re-run: skips any
# file that's already downloaded. No Python/ultralytics required —
# this hits the GitHub release assets directly with curl/wget.
#
# Usage:
#   ./scripts/download_models.sh
#
# This is Phase 1 of the build roadmap: today it's just YOLOv8-nano.
# Depth-Anything / Whisper / sentence-transformers are pulled
# automatically by their own libraries on first use in later phases,
# so they're intentionally not duplicated here.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Resolve paths relative to this script's location, not the caller's cwd,
# so it works whether you run it from project root or from inside scripts/.
# Mirrors MODELS_DIR in backend/app/config.py exactly — don't let these drift.
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/data/models"

mkdir -p "${MODELS_DIR}"
echo "==> Models directory: ${MODELS_DIR}"

# --------------------------------------------------------------------------- #
# Downloader: curl preferred, wget as fallback.
# --------------------------------------------------------------------------- #
download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 10 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --tries=3 -O "${out}" "${url}"
  else
    echo "ERROR: neither curl nor wget is installed." >&2
    exit 1
  fi
}

# --------------------------------------------------------------------------- #
# Resolve the latest ultralytics/assets release tag so this script keeps
# working as new releases ship. Falls back to a pinned, known-good tag
# if the API call fails (offline, rate-limited, etc.) or the asset is
# missing from latest (newer releases sometimes drop legacy weight names).
# --------------------------------------------------------------------------- #
PINNED_FALLBACK_RELEASE="v8.3.0"

resolve_latest_release() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://api.github.com/repos/ultralytics/assets/releases/latest" 2>/dev/null \
      | grep -m1 '"tag_name"' \
      | sed -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/' || true
  fi
}

# --------------------------------------------------------------------------- #
# YOLOv8-nano (Module 1 — object detection, ~6MB)
# --------------------------------------------------------------------------- #
fetch_yolov8n() {
  local filename="yolov8n.pt"
  local target="${MODELS_DIR}/${filename}"
  local base_url="https://github.com/ultralytics/assets/releases/download"

  if [[ -f "${target}" && -s "${target}" ]]; then
    echo "==> ${filename} already present, skipping. (delete it to force re-download)"
    return 0
  fi

  local tmp_file="${target}.part"
  local latest_tag
  latest_tag="$(resolve_latest_release)"

  local tried_tags=()
  [[ -n "${latest_tag}" ]] && tried_tags+=("${latest_tag}")
  tried_tags+=("${PINNED_FALLBACK_RELEASE}")

  local ok=0
  for tag in "${tried_tags[@]}"; do
    echo "==> Trying ${filename} from release ${tag}..."
    if download "${base_url}/${tag}/${filename}" "${tmp_file}"; then
      # Sanity check: must be a real binary checkpoint, not an HTML error page
      # from a bad redirect, and big enough to plausibly be the real weights.
      if ! file "${tmp_file}" 2>/dev/null | grep -qi "html" \
         && [[ $(wc -c < "${tmp_file}") -gt 1000000 ]]; then
        ok=1
        break
      fi
      echo "    -> response didn't look like a valid checkpoint, trying next source."
      rm -f "${tmp_file}"
    fi
  done

  if [[ "${ok}" -ne 1 ]]; then
    echo "ERROR: failed to download ${filename} from any known release." >&2
    echo "       Check your network connection, or download it manually from:" >&2
    echo "       https://github.com/ultralytics/assets/releases" >&2
    rm -f "${tmp_file}"
    exit 1
  fi

  mv "${tmp_file}" "${target}"
  local size_human
  size_human="$(du -h "${target}" | cut -f1)"
  echo "==> Saved ${filename} (${size_human}) to ${target}"
}

fetch_yolov8n

echo
echo "==> Done. Model path matches settings.YOLO_MODEL_PATH in backend/app/config.py"
echo "    — no config changes needed."