#!/usr/bin/env bash
# SentenceMaker installation script (idempotent)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-sentencemaker}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="$PROJECT_ROOT/venv"

log() {
    printf "\033[1;34m[install]\033[0m %s\n" "$*"
}

warn() {
    printf "\033[1;33m[install]\033[0m %s\n" "$*" >&2
}

die() {
    printf "\033[0;31m[install]\033[0m %s\n" "$*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ensure_python_range() {
    python3 <<'PYCHECK'
import sys
major, minor = sys.version_info[:2]
if major != 3 or not (8 <= minor <= 12):
    raise SystemExit(
        "Python 3.8-3.12 required for venv installation; install conda or a compatible Python."
    )
PYCHECK
}

declare -a PYTHON_CMD
USE_CONDA=0

if command_exists conda; then
    USE_CONDA=1
    log "Conda detected. Target environment: $ENV_NAME (Python $PYTHON_VERSION)."
    if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
        log "Conda environment '$ENV_NAME' already exists; reusing."
    else
        log "Creating conda environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi
    PYTHON_CMD=(conda run -n "$ENV_NAME" python)
else
    log "Conda not found. Falling back to virtualenv at $VENV_DIR."
    command_exists python3 || die "python3 not found. Install Python 3.8-3.12 or conda, then retry."
    ensure_python_range
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        log "Virtual environment already exists; reusing."
    fi
    PYTHON_CMD=("$VENV_DIR/bin/python")
fi

run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

run_pip() {
    "${PYTHON_CMD[@]}" -m pip "$@"
}

log "Upgrading pip/setuptools/wheel..."
run_pip install --upgrade pip setuptools wheel

if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
    die "requirements.txt not found in $PROJECT_ROOT"
fi

log "Installing Python dependencies from requirements.txt..."
run_pip install -r "$PROJECT_ROOT/requirements.txt"

log "Ensuring spaCy language models are available..."
run_python -m spacy download es_core_news_sm
run_python -m spacy download es_core_news_md

log "Installation complete."
if (( USE_CONDA )); then
    log "Activate the environment with: conda activate $ENV_NAME"
else
    log "Activate the environment with: source $VENV_DIR/bin/activate"
fi

log "Next step: run ./start.sh to launch SentenceMaker."
