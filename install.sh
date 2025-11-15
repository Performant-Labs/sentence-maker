#!/usr/bin/env bash
# Automated macOS installer for SentenceMaker

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
CONDA_ENV_NAME="sentencemaker"
PYTHON_VERSION="3.12"

fmt_info() { printf "\033[1;34m==>\033[0m %s\n" "$1"; }
fmt_warn() { printf "\033[1;33m==>\033[0m %s\n" "$1"; }
fmt_error() { printf "\033[1;31m==>\033[0m %s\n" "$1" >&2; }

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        fmt_error "This installer is intended for macOS only."
        exit 1
    fi
}

require_homebrew() {
    if ! command -v brew >/dev/null 2>&1; then
        fmt_error "Homebrew not found. Install it from https://brew.sh and re-run this script."
        exit 1
    fi
}

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        return
    fi

    fmt_warn "Conda not detected. Installing Miniconda via Homebrew (admin prompt may appear)..."
    brew install --cask miniconda

    if [[ -x "/usr/local/Caskroom/miniconda/latest/base/bin/conda" ]]; then
        CONDA_BIN="/usr/local/Caskroom/miniconda/latest/base/bin/conda"
    elif [[ -x "/opt/homebrew/Caskroom/miniconda/latest/base/bin/conda" ]]; then
        CONDA_BIN="/opt/homebrew/Caskroom/miniconda/latest/base/bin/conda"
    else
        fmt_error "Unable to locate conda after installation. Please ensure Miniconda is on your PATH."
        exit 1
    fi

    eval "$("$CONDA_BIN" shell.bash hook)"
}

create_conda_env() {
    if "$CONDA_BIN" env list | awk 'NR>2 {print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        fmt_info "Conda environment '$CONDA_ENV_NAME' already exists."
    else
        fmt_info "Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
        "$CONDA_BIN" create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
    fi
}

conda_run() {
    "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" "$@"
}

install_python_deps() {
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        fmt_error "requirements.txt not found at $REQUIREMENTS_FILE"
        exit 1
    fi

    fmt_info "Upgrading pip/setuptools/wheel in '$CONDA_ENV_NAME'..."
    conda_run python -m pip install --upgrade pip setuptools wheel

    fmt_info "Installing Python dependencies..."
    conda_run python -m pip install -r "$REQUIREMENTS_FILE"

    fmt_info "Ensuring spaCy Spanish model is installed..."
    if ! conda_run python -m spacy validate 2>/dev/null | grep -q "es_core_news_sm"; then
        conda_run python -m spacy download es_core_news_sm
    else
        fmt_info "spaCy Spanish model already available."
    fi
}

ensure_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        fmt_info "Ollama already installed."
        return
    fi
    fmt_info "Installing Ollama (this may prompt for admin approval)..."
    brew install --cask ollama
    fmt_warn "Launch the Ollama app once if prompted to finish setup."
}

pull_ollama_models() {
    if ! command -v ollama >/dev/null 2>&1; then
        fmt_warn "Ollama CLI not available; skipping model download."
        return
    fi

    if ! ollama list >/dev/null 2>&1; then
        fmt_warn "Ollama service is not running. Start the Ollama app, then run 'ollama pull gemma2:9b'."
        return
    fi

    if ! ollama list | grep -q "gemma2:9b"; then
        fmt_info "Downloading gemma2:9b model (required)..."
        ollama pull gemma2:9b
    else
        fmt_info "gemma2:9b already downloaded."
    fi

    fmt_info "Optional models: run 'ollama pull gemma2:27b' or 'ollama pull mistral:7b-instruct-v0.2-q4_0' later if desired."
}

print_summary() {
    cat <<EOF

Installation complete!
Next steps:
  1. Activate the environment: conda activate $CONDA_ENV_NAME
  2. Ensure the Ollama service is running (start the app if needed).
  3. Launch SentenceMaker with: ./start.sh

Optional models (install later if needed):
  ollama pull gemma2:27b              # Highest quality
  ollama pull mistral:7b-instruct-v0.2-q4_0  # Faster alternative
EOF
}

main() {
    require_macos
    require_homebrew
    ensure_conda
    create_conda_env
    install_python_deps
    ensure_ollama
    pull_ollama_models
    print_summary
}

main "$@"
