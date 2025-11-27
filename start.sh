#!/bin/bash
# SentenceMaker Startup Script
# Ensures clean environment and starts the sentence generator

set -e  # Exit on error

# ============================================================================
# CONFIGURATION: Default Values
# ============================================================================

# Default LLM Model
# The model used for sentence generation and validation.
# Options:
#   - qwen2.5:14b   : Recommended (best balance, ~10GB RAM)
#   - qwen2.5:32b   : Highest quality (~24GB RAM, may use swap on 32GB systems)
#   - gemma2:9b     : Fastest (~8GB RAM, lower quality)
#   - gemma2:27b    : Good quality (~20GB RAM)
DEFAULT_LLM_MODEL="qwen2.5:14b"
# LLM Providers (generation vs validation defaults)
DEFAULT_GEN_LLM_PROVIDER="openai"
DEFAULT_VAL_LLM_PROVIDER="none"
# Default LLM model for validation (can be lighter than generation model)
DEFAULT_VALIDATOR_LLM_MODEL="gemma2:9b"
# API keys (stored in a separate, ignored file). Exported env vars override file values.
KEY_FILE=".sentencemaker_keys"
if [ -f "$KEY_FILE" ]; then
    # shellcheck disable=SC1090
    . "$KEY_FILE"
fi
export OPENAI_API_KEY
export ANTHROPIC_API_KEY

# Ollama Server Configuration
OLLAMA_PORT="11434"                    # Default Ollama API port
OLLAMA_URL="http://localhost:${OLLAMA_PORT}"
OLLAMA_LOG_FILE="/tmp/ollama.log"      # Log file for Ollama server output

# Test Generation Settings
# Used to verify Ollama is responsive before starting
TEST_TIMEOUT_SECONDS=10                # Max time to wait for test generation
TEST_NUM_TOKENS=5                      # Number of tokens to generate in test
SLOW_GENERATION_THRESHOLD=5            # Warn if test takes longer than this (seconds)

# Startup Timeouts
OLLAMA_STARTUP_TIMEOUT=15              # Max seconds to wait for Ollama to start
GRACEFUL_SHUTDOWN_TIMEOUT=5            # Seconds to wait for graceful process shutdown

# File Paths
LAST_RUN_FILE=".sentencemaker_last_command"  # Stores last command for resume
CHECKPOINT_PATTERN="*.state.json"      # Pattern to find checkpoint files
MAX_CHECKPOINT_DEPTH=5                 # Max directory depth to search for checkpoints

# Default Generation Parameters (used by wizard)
DEFAULT_WORDLIST="words/words.txt"
DEFAULT_OUTPUT="output/sentences.txt"
DEFAULT_MAX_WORDS=12
DEFAULT_MIN_WORDS=6
DEFAULT_MAX_SENTENCES=0                # 0 = generate until all words are used

# ============================================================================
# END OF CONFIGURATION
# ============================================================================
# You can customize any values above to change default behavior.
# Most important settings:
#   - DEFAULT_LLM_MODEL: Change the default model (qwen2.5:14b recommended)
#   - DEFAULT_GEN_LLM_PROVIDER: Provider to use for generation (ollama = local/free, API providers incur cost)
#   - DEFAULT_VALIDATOR_LLM_MODEL: Set a lighter/different model for validation
#   - DEFAULT_MAX_WORDS: Adjust maximum words per sentence
#   - OLLAMA_STARTUP_TIMEOUT: Increase if Ollama takes longer to start
#   - TEST_TIMEOUT_SECONDS: Increase if test generations timeout
# ============================================================================

# ============================================================================
# CONFIGURATION: Colors for Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Function to show help
show_help() {
    echo "============================================================"
    echo "SentenceMaker Startup Script"
    echo "============================================================"
    echo ""
    echo "Usage: ./start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)          Normal startup with cleanup and wizard"
    echo "  --reset            Force restart Ollama server"
    echo "  --diagnostics      Show system diagnostics and exit"
    echo "  --diag             Alias for --diagnostics"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh                 # Normal startup"
    echo "  ./start.sh --reset         # Reset Ollama and start"
    echo "  ./start.sh --diagnostics   # Check system status"
    echo ""
    exit 0
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

echo "============================================================"
echo "SentenceMaker Startup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
LAST_RUN_FILE=".sentencemaker_last_command"

# Python detection helper
PYTHON_BIN=""
detect_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
        return 0
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python)
        return 0
    fi
    return 1
}

# Function to show diagnostic information
show_diagnostics() {
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW}System Diagnostics${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    echo ""

    # Python processes
    echo -e "${YELLOW}Python Processes:${NC}"
    local PYTHON_PROCS=$(ps aux | grep -E "python.*[Ss]entence" | grep -v grep || true)
    if [ -n "$PYTHON_PROCS" ]; then
        echo "$PYTHON_PROCS"
    else
        echo "  None found"
    fi
    echo ""

    # Ollama processes
    echo -e "${YELLOW}Ollama Processes:${NC}"
    local OLLAMA_PROCS=$(ps aux | grep ollama | grep -v grep || true)
    if [ -n "$OLLAMA_PROCS" ]; then
        echo "$OLLAMA_PROCS"
    else
        echo "  None found"
    fi
    echo ""

    # Ollama runners
    echo -e "${YELLOW}Ollama Runners:${NC}"
    local RUNNER_PROCS=$(pgrep -f "ollama runner" || true)
    if [ -n "$RUNNER_PROCS" ]; then
        for pid in $RUNNER_PROCS; do
            local INFO=$(ps -p $pid -o pid=,ppid=,etime=,command= 2>/dev/null || echo "")
            if [ -n "$INFO" ]; then
                echo "  $INFO"
            fi
        done
    else
        echo "  None found"
    fi
    echo ""

    # Port usage
    echo -e "${YELLOW}Port Usage:${NC}"
    for port in 11434 5000 8000 8080 3000; do
        local PORT_INFO=$(lsof -nP -iTCP:$port -sTCP:LISTEN 2>/dev/null || true)
        if [ -n "$PORT_INFO" ]; then
            echo "  Port $port:"
            echo "$PORT_INFO" | tail -n +2 | awk '{print "    PID: "$2", Command: "$1}'
        fi
    done
    echo ""

    # Ollama responsiveness
    echo -e "${YELLOW}Ollama Server Status:${NC}"
    if curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Responsive on port 11434${NC}"

        # Test generation with timing
        echo -n "  Testing generation speed... "
        local gen_start=$(date +%s.%N)
        if curl -s --max-time ${TEST_TIMEOUT_SECONDS} http://localhost:${OLLAMA_PORT}/api/generate -d '{"model": "'${DEFAULT_LLM_MODEL}'", "prompt": "Test", "stream": false, "options": {"num_predict": '${TEST_NUM_TOKENS}'}}' 2>&1 | grep -q "response"; then
            local gen_end=$(date +%s.%N)
            local gen_time=$(echo "$gen_end - $gen_start" | bc)
            echo -e "${GREEN}✓ Working (${gen_time}s)${NC}"

            # Warn if slow
            if (( $(echo "$gen_time > ${SLOW_GENERATION_THRESHOLD}" | bc -l) )); then
                echo -e "  ${YELLOW}⚠ Generation is slow (>${SLOW_GENERATION_THRESHOLD}s for ${TEST_NUM_TOKENS} tokens)${NC}"
                echo -e "  ${YELLOW}  This may indicate high system load or memory pressure${NC}"
            fi
        else
            echo -e "${RED}✗ Generation not working${NC}"
        fi
    else
        echo -e "  ${RED}✗ Not responsive${NC}"
    fi
    echo ""

    # System resources
    echo -e "${YELLOW}System Resources:${NC}"
    local mem_info=$(top -l 1 -n 0 | grep PhysMem)
    echo "  $mem_info"
    local ollama_mem=$(ps aux | grep "ollama" | grep -v grep | awk '{sum+=$6} END {printf "%.1f", sum/1024}')
    if [ -n "$ollama_mem" ] && [ "$ollama_mem" != "0.0" ]; then
        echo "  Ollama memory usage: ${ollama_mem} MB"
    fi
    echo ""

    # Temp files
    echo -e "${YELLOW}Temporary Files:${NC}"
    ls -lh /tmp/ollama.log 2>/dev/null || echo "  No ollama.log"
    ls -lh /tmp/sentencemaker_* 2>/dev/null || true
    echo ""

    echo -e "${YELLOW}============================================================${NC}"
    exit 0
}

detect_checkpoint_files() {
    local files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$PWD" -maxdepth ${MAX_CHECKPOINT_DEPTH} -name "${CHECKPOINT_PATTERN}" ! -path "./venv/*" -print0)
    echo "${files[@]}"
}

resume_previous_run() {
    local state_files=("$@")
    local response
    echo -e "\n${YELLOW}Detected unfinished run (checkpoint file found).${NC}"
    for file in "${state_files[@]}"; do
        echo "  - $file"
    done
    read -r -p "Resume with previous settings? [Y/n]: " response
    response=${response:-y}
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Starting fresh. Removing checkpoints."
        for file in "${state_files[@]}"; do
            rm -f "$file"
        done
        rm -f "$LAST_RUN_FILE" 2>/dev/null || true
        return 1
    fi

    if [[ ! -f "$LAST_RUN_FILE" ]]; then
        echo -e "${RED}Cannot resume automatically: previous command not recorded.${NC}"
        return 1
    fi

    local last_cmd
    last_cmd=$(cat "$LAST_RUN_FILE")
    echo -e "${GREEN}Resuming with:${NC} $last_cmd"
    eval "$last_cmd"
    exit 0
}

# Check for flags
if [[ "$1" == "--diagnostics" ]] || [[ "$1" == "--diag" ]]; then
    show_diagnostics
fi

# Check for --reset flag
if [[ "$1" == "--reset" ]]; then
    echo -e "${YELLOW}Resetting Ollama...${NC}"

    # Kill all Ollama processes (including runners)
    echo "  Stopping all Ollama processes..."
    pkill -TERM ollama 2>/dev/null || true

    # Wait up to 5 seconds for graceful shutdown
    for i in {1..5}; do
        if ! pgrep ollama >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓ Ollama stopped gracefully${NC}"
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if pgrep ollama >/dev/null 2>&1; then
        echo "  Forcing shutdown..."
        pkill -9 ollama 2>/dev/null || true
        sleep 1
        echo -e "  ${GREEN}✓ Ollama force stopped${NC}"
    fi

    # Clean up temp files
    rm -f /tmp/ollama.log 2>/dev/null || true

    # Restart Ollama
    echo "  Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "  Ollama PID: $OLLAMA_PID"
    sleep 5

    # Verify it's running
    if pgrep ollama >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Ollama restarted successfully${NC}"

        # Test responsiveness
        echo "  Testing Ollama..."
        if curl -s --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓ Ollama is responsive${NC}"
        else
            echo -e "  ${YELLOW}⚠ Ollama started but not responding yet (may need more time)${NC}"
        fi
        echo ""
    else
        echo -e "  ${RED}✗ Failed to restart Ollama${NC}"
        echo "  Check logs: tail -f /tmp/ollama.log"
        exit 1
    fi
fi

# Function to detect Ollama port
get_ollama_port() {
    # Try to detect port from running process, but filter out runner ports (>50000)
    local port=$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep "ollama" | grep -v "ollama runner" | grep -oE ':[0-9]+' | head -1 | tr -d ':')

    # Default to 11434 if not found
    if [ -z "$port" ]; then
        port="11434"
    fi

    echo "$port"
}

# Function to check if ollama is running and responsive
check_ollama() {
    local OLLAMA_PORT=$(get_ollama_port)
    local OLLAMA_URL="http://localhost:${OLLAMA_PORT}"

    # First check if API responds
    if ! curl -s --max-time 5 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        return 1
    fi

    # Test actual generation to ensure it's not stuck
    echo -e "${YELLOW}Testing Ollama responsiveness (port ${OLLAMA_PORT})...${NC}"
    local test_response=$(curl -s --max-time ${TEST_TIMEOUT_SECONDS} "${OLLAMA_URL}/api/generate" -d '{
        "model": "'${DEFAULT_LLM_MODEL}'",
        "prompt": "Test",
        "stream": false,
        "options": {"num_predict": '${TEST_NUM_TOKENS}'}
    }' 2>&1)

    if [ $? -eq 0 ] && echo "$test_response" | grep -q "response"; then
        echo -e "${GREEN}✓ Ollama is responsive${NC}"
        return 0
    else
        echo -e "${RED}✗ Ollama is not responding properly${NC}"
        return 1
    fi
}

# Function to stop hung processes
cleanup_processes() {
    echo -e "${YELLOW}Step 1: Cleaning up any hung processes...${NC}"

    local found_processes=0

    # 1. Clean up Python processes related to this project
    echo "  Checking for Python processes..."
    local PYTHON_PIDS=$(pgrep -f "sentencemaker\.py|sentence_generator\.py|llm_validator\.py|agreement_validator\.py|word_classifier\.py|word_transformer\.py" || true)

    if [ -n "$PYTHON_PIDS" ]; then
        found_processes=1
        echo "    Found Python processes: $PYTHON_PIDS"
        echo "    Attempting graceful shutdown (SIGTERM)..."
        for pid in $PYTHON_PIDS; do
            kill -TERM "$pid" 2>/dev/null || true
        done

        # Wait up to 5 seconds for graceful exit
        for i in {1..5}; do
            REMAINING=$(pgrep -f "sentencemaker\.py|sentence_generator\.py|llm_validator\.py|agreement_validator\.py|word_classifier\.py|word_transformer\.py" || true)
            if [ -z "$REMAINING" ]; then
                echo -e "    ${GREEN}✓ Python processes stopped gracefully${NC}"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        REMAINING=$(pgrep -f "sentencemaker\.py|sentence_generator\.py|llm_validator\.py|agreement_validator\.py|word_classifier\.py|word_transformer\.py" || true)
        if [ -n "$REMAINING" ]; then
            echo "    Forcing shutdown (SIGKILL)..."
            for pid in $REMAINING; do
                kill -KILL "$pid" 2>/dev/null || true
            done
            sleep 1
            echo -e "    ${GREEN}✓ Python processes forcefully stopped${NC}"
        fi
    fi

    # 2. Clean up orphaned Ollama runner processes
    echo "  Checking for orphaned Ollama runners..."
    local RUNNER_PIDS=$(pgrep -f "ollama runner" || true)

    if [ -n "$RUNNER_PIDS" ]; then
        found_processes=1
        echo "    Found orphaned runners: $RUNNER_PIDS"
        echo "    Terminating runners..."
        for pid in $RUNNER_PIDS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
        echo -e "    ${GREEN}✓ Ollama runners cleaned up${NC}"
    fi

    # 3. Clean up stuck Ollama server (if unresponsive)
    echo "  Checking for stuck Ollama server..."
    if pgrep -x "ollama" >/dev/null 2>&1; then
        # Check if it's responsive
        if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
            found_processes=1
            echo "    Found unresponsive Ollama server"
            echo "    Terminating Ollama..."
            pkill -TERM ollama 2>/dev/null || true
            sleep 2

            # Force kill if still running
            if pgrep -x "ollama" >/dev/null 2>&1; then
                pkill -KILL ollama 2>/dev/null || true
                sleep 1
            fi
            echo -e "    ${GREEN}✓ Ollama server cleaned up${NC}"
        fi
    fi

    # 4. Check for processes holding common ports
    echo "  Checking for port conflicts..."
    local PORT_CONFLICTS=0
    for port in 11434 5000 8000 8080 3000; do
        local PORT_PID=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$PORT_PID" ]; then
            local PORT_CMD=$(ps -p $PORT_PID -o comm= 2>/dev/null || echo "unknown")
            # Only kill if it's related to our project
            if [[ "$PORT_CMD" =~ (python|ollama) ]]; then
                found_processes=1
                PORT_CONFLICTS=1
                echo "    Port $port occupied by PID $PORT_PID ($PORT_CMD)"
                kill -TERM "$PORT_PID" 2>/dev/null || true
            fi
        fi
    done

    if [ $PORT_CONFLICTS -eq 1 ]; then
        sleep 2
        echo -e "    ${GREEN}✓ Port conflicts resolved${NC}"
    fi

    # 5. Clean up temporary files and locks
    echo "  Cleaning temporary files..."
    rm -f /tmp/ollama.log 2>/dev/null || true
    rm -f /tmp/sentencemaker_*.lock 2>/dev/null || true
    rm -f /tmp/sentencemaker_*.pid 2>/dev/null || true

    if [ $found_processes -eq 0 ]; then
        echo -e "  ${GREEN}✓ No hung processes found${NC}"
    else
        echo -e "  ${GREEN}✓ Cleanup complete${NC}"
    fi
}

# Function to start ollama if not running
start_ollama() {
    echo -e "\n${YELLOW}Step 2: Checking ollama server...${NC}"

    # First check if it's already running and responsive
    if check_ollama; then
        echo -e "  ${GREEN}✓ Ollama is already running and responsive${NC}"
        return 0
    fi

    # Check if ollama process exists but is unresponsive
    if pgrep -x "ollama" >/dev/null 2>&1; then
        echo "  Ollama process found but unresponsive, restarting..."
        pkill -TERM ollama 2>/dev/null || true
        sleep 2

        # Force kill if still running
        if pgrep -x "ollama" >/dev/null 2>&1; then
            pkill -KILL ollama 2>/dev/null || true
            sleep 1
        fi
    fi

    echo "  Starting ollama server..."

    # Start ollama in background with logging
    nohup ollama serve > ${OLLAMA_LOG_FILE} 2>&1 &
    OLLAMA_PID=$!
    echo "  Ollama PID: $OLLAMA_PID"

    # Wait for ollama to be ready (up to ${OLLAMA_STARTUP_TIMEOUT} seconds with progress)
    for i in $(seq 1 ${OLLAMA_STARTUP_TIMEOUT}); do
        if check_ollama; then
            echo -e "  ${GREEN}✓ Ollama started successfully${NC}"
            return 0
        fi

        # Show progress every 3 seconds
        if [ $((i % 3)) -eq 0 ]; then
            echo "  Waiting for Ollama to be ready... (${i}s)"
        fi
        sleep 1
    done

    echo -e "  ${RED}✗ Failed to start ollama${NC}"
    echo "  Check logs: tail -f /tmp/ollama.log"
    return 1
}

# Function to check conda environment
check_environment() {
    echo -e "\n${YELLOW}Step 3: Checking Python environment...${NC}"

    # Check if in sentencemaker environment
    if [[ "$CONDA_DEFAULT_ENV" == "sentencemaker" ]]; then
        echo -e "  ${GREEN}✓ sentencemaker environment is active${NC}"
        if detect_python; then
            "$PYTHON_BIN" --version
        else
            echo -e "  ${RED}✗ Python 3 executable not found${NC}"
            return 1
        fi
        return 0
    fi

    echo -e "  ${YELLOW}⚠ sentencemaker environment not active${NC}"
    echo "  Attempting to activate..."

    # Try to activate
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate sentencemaker 2>/dev/null || {
            echo -e "  ${RED}✗ Failed to activate sentencemaker environment${NC}"
            echo "  Please run: conda activate sentencemaker"
            return 1
        }
        echo -e "  ${GREEN}✓ Environment activated${NC}"
        if detect_python; then
            "$PYTHON_BIN" --version
        else
            echo -e "  ${RED}✗ Python 3 executable not found${NC}"
            return 1
        fi
    else
        echo -e "  ${RED}✗ Conda not found${NC}"
        return 1
    fi
}

# Function to verify ollama models
check_models() {
    echo -e "\n${YELLOW}Step 4: Checking ollama models...${NC}"

    if ! check_ollama; then
        echo -e "  ${YELLOW}⚠ Ollama not running, skipping model check${NC}"
        return 0
    fi

    # Check for recommended models
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || true)

    if [ -z "$MODELS" ]; then
        echo -e "  ${YELLOW}⚠ No models found${NC}"
        echo "  To use LLM filtering, pull a model:"
        echo "    ollama pull ${DEFAULT_LLM_MODEL}   # Recommended default (best balance)"
        echo "    ollama pull qwen2.5:32b   # Highest quality (requires more RAM)"
        echo "    ollama pull gemma2:9b     # Fastest (lower quality)"
        return 0
    fi

    echo "  Available models:"
    # Highlight recommended models
    while IFS= read -r model; do
        if [[ "$model" == "qwen2.5:14b" ]]; then
            echo -e "    • $model ${GREEN}(recommended default - best balance)${NC}"
        elif [[ "$model" == "qwen2.5:32b" ]]; then
            echo -e "    • $model ${YELLOW}(highest quality, more RAM)${NC}"
        elif [[ "$model" == "gemma2:27b" ]]; then
            echo -e "    • $model ${YELLOW}(good quality)${NC}"
        elif [[ "$model" == "gemma2:9b" ]]; then
            echo -e "    • $model ${YELLOW}(fastest)${NC}"
        elif [[ "$model" == "mistral:7b-instruct"* ]]; then
            echo -e "    • $model ${YELLOW}(faster)${NC}"
        else
            echo "    • $model"
        fi
    done <<< "$MODELS"
    echo -e "  ${GREEN}✓ Models ready${NC}"
}

# Function to run configuration wizard
run_wizard() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}Configuration Wizard${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "Let's configure your sentence generation run."
    echo "Press Enter to use the default value shown in [brackets]."
    echo ""

    # Word list
    echo -n "Word list file [${DEFAULT_WORDLIST}]: "
    read -r wordlist
    wordlist=${wordlist:-${DEFAULT_WORDLIST}}

    # Output file
    echo -n "Output file [${DEFAULT_OUTPUT}]: "
    read -r output
    output=${output:-${DEFAULT_OUTPUT}}

    # Max words
    echo -n "Maximum words per sentence [${DEFAULT_MAX_WORDS}]: "
    read -r max_words
    max_words=${max_words:-${DEFAULT_MAX_WORDS}}

    echo -n "Minimum words per sentence (after fixes) [${DEFAULT_MIN_WORDS}]: "
    read -r min_words
    min_words=${min_words:-${DEFAULT_MIN_WORDS}}

    echo -n "Maximum sentences to generate (0 = until done) [${DEFAULT_MAX_SENTENCES}]: "
    read -r max_sentences
    max_sentences=${max_sentences:-${DEFAULT_MAX_SENTENCES}}
    # Normalize stray characters (e.g., accidental trailing backslash) and enforce numeric
    max_sentences=${max_sentences//\\/}
    if ! [[ "$max_sentences" =~ ^[0-9]+$ ]]; then
        echo -e "${YELLOW}Warning:${NC} Invalid max sentences '${max_sentences}', reverting to ${DEFAULT_MAX_SENTENCES}"
        max_sentences=${DEFAULT_MAX_SENTENCES}
    fi

    echo ""
    echo -e "${YELLOW}LLM Model Selection:${NC}"
    echo "LLM generates semantically coherent sentences (required)."
    echo ""

    # Predefined API model lists (letters distinguish paid API from local numeric selection)
    OPENAI_HIGH=("gpt-5.1" "gpt-5.1-chat-latest" "gpt-5-pro")
    OPENAI_SECOND=("gpt-4.1" "gpt-4o" "gpt-4o-mini")
    ANTHROPIC_HIGH=("claude-3.5-sonnet-20241022" "claude-3.5-sonnet" "claude-3-opus-20240229")
    ANTHROPIC_SECOND=("claude-3-haiku-20240307" "claude-3-haiku" "claude-3-sonnet")

    choose_api_model_inline() {
        local provider="$1"
        local -a high_list=("${!2}")
        local -a second_list=("${!3}")
        local key_in_use=""
        if [[ "$provider" == "OpenAI" ]]; then
            key_in_use="$OPENAI_API_KEY"
        else
            key_in_use="$ANTHROPIC_API_KEY"
        fi
        local key_snippet="${key_in_use:0:6}"
        printf "Using %s API key (first 6 chars): %s\n" "${provider}" "${key_snippet:-<none>}" >&2
        printf "Available %s models (letters = paid API):\n" "${provider}" >&2
        local idx=1
        for m in "${high_list[@]}"; do
            printf "  A%d. %s (highest tier)\n" "$idx" "$m" >&2
            ((idx++))
        done
        idx=1
        for m in "${second_list[@]}"; do
            printf "  B%d. %s (second tier)\n" "$idx" "$m" >&2
            ((idx++))
        done
        printf "Select model code [A1]: " >&2
        read -r choice
        choice=${choice:-"A1"}
        local selected=""
        if [[ "$choice" =~ ^A([1-3])$ ]]; then
            local pos=${BASH_REMATCH[1]}
            selected="${high_list[$((pos-1))]}"
        elif [[ "$choice" =~ ^B([1-3])$ ]]; then
            local pos=${BASH_REMATCH[1]}
            selected="${second_list[$((pos-1))]}"
        else
            selected="${high_list[0]}"
        fi
        echo "$selected"
    }

    # Generation provider/model
    echo -n "Generation provider [ollama/openai/anthropic] [${DEFAULT_GEN_LLM_PROVIDER}]: "
    read -r gen_provider
    gen_provider=${gen_provider:-${DEFAULT_GEN_LLM_PROVIDER}}

    selected_model="${DEFAULT_LLM_MODEL}"
    selected_validator_model="${DEFAULT_VALIDATOR_LLM_MODEL}"

    # If generation uses Ollama, ensure responsive and list models
    if [[ "$gen_provider" == "ollama" ]]; then
        if ! check_ollama; then
            echo ""
            echo -e "${RED}Cannot continue without a responsive Ollama server.${NC}"
            exit 1
        fi
        echo ""
        echo "Available Ollama models:"
        MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || true)
        if [ -n "$MODELS" ]; then
            i=1
            default_num=""
            declare -a model_array
            while IFS= read -r model; do
                model_array[$i]=$model
                if [[ "$model" == "$DEFAULT_LLM_MODEL" ]]; then
                    echo -e "  $i. $model ${GREEN}(default generation)${NC}"
                    default_num=$i
                elif [[ "$model" == "$DEFAULT_VALIDATOR_LLM_MODEL" ]]; then
                    echo -e "  $i. $model ${YELLOW}(default validator)${NC}"
                elif [[ "$model" == "qwen2.5:32b" ]]; then
                    echo -e "  $i. $model ${YELLOW}(highest quality, more RAM)${NC}"
                elif [[ "$model" == "gemma2:27b" ]]; then
                    echo -e "  $i. $model ${YELLOW}(good quality)${NC}"
                elif [[ "$model" == "gemma2:9b" ]]; then
                    echo -e "  $i. $model ${YELLOW}(fastest)${NC}"
                elif [[ "$model" == "mistral:7b-instruct"* ]]; then
                    echo -e "  $i. $model ${YELLOW}(faster)${NC}"
                else
                    echo "  $i. $model"
                fi
                ((i++))
            done <<< "$MODELS"

            echo ""
            if [ -n "$default_num" ]; then
        echo -n "Select generation model number [$default_num for ${model_array[$default_num]}]: "
        read -r model_num
        model_num=${model_num:-$default_num}
    else
        echo "Available Ollama models:"
        i=1
        for model in "${model_array[@]}"; do
            [ -z "$model" ] && continue
            echo "  $i. $model"
            ((i++))
        done
        echo -n "Select generation model number [1]: "
        read -r model_num
        model_num=${model_num:-1}
    fi

            if [ -n "${model_array[$model_num]}" ]; then
                selected_model=${model_array[$model_num]}
            fi
        else
            echo -e "${YELLOW}No Ollama models found. Using default: ${selected_model}${NC}"
        fi
    elif [[ "$gen_provider" == "openai" ]]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${RED}OPENAI_API_KEY not set. Please add it to $KEY_FILE or export it.${NC}"
            exit 1
        fi
        selected_model=$(choose_api_model_inline "OpenAI" OPENAI_HIGH[@] OPENAI_SECOND[@])
    else
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo -e "${RED}ANTHROPIC_API_KEY not set. Please add it to $KEY_FILE or export it.${NC}"
            exit 1
        fi
        selected_model=$(choose_api_model_inline "Anthropic" ANTHROPIC_HIGH[@] ANTHROPIC_SECOND[@])
    fi

    # Validation provider/model
    echo ""
    echo -n "Validation provider [none/ollama/openai/anthropic] [${DEFAULT_VAL_LLM_PROVIDER}]: "
    read -r val_provider
    val_provider=${val_provider:-${DEFAULT_VAL_LLM_PROVIDER}}

    if [[ "$val_provider" == "none" ]]; then
        selected_validator_model="none"
    elif [[ "$val_provider" == "ollama" ]]; then
        if ! check_ollama; then
            echo ""
            echo -e "${RED}Cannot continue without a responsive Ollama server.${NC}"
            exit 1
        fi
        selected_validator_model="${DEFAULT_VALIDATOR_LLM_MODEL}"
        if [ -n "$MODELS" ]; then
            echo ""
            echo -e "${YELLOW}Validator LLM Selection:${NC}"
            echo "Validator checks coherence; can use a smaller model for speed."
            # reuse model_array if present
            if [ ${#model_array[@]} -eq 0 ]; then
                i=1
                declare -a model_array
                while IFS= read -r model; do
                    model_array[$i]=$model
                    ((i++))
                done <<< "$MODELS"
            fi
            local_validator_default=1
            echo "Available Ollama models for validation:"
            j=1
            for model in "${model_array[@]}"; do
                [ -z "$model" ] && continue
                echo "  $j. $model"
                ((j++))
            done
            echo -n "Select validator model number [$local_validator_default for ${model_array[$local_validator_default]}]: "
            read -r validator_num
            validator_num=${validator_num:-$local_validator_default}
            if [ -n "${model_array[$validator_num]}" ]; then
                selected_validator_model=${model_array[$validator_num]}
            fi
        fi
    elif [[ "$val_provider" == "openai" ]]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${RED}OPENAI_API_KEY not set. Please add it to $KEY_FILE or export it.${NC}"
            exit 1
        fi
        selected_validator_model=$(choose_api_model_inline "OpenAI" OPENAI_HIGH[@] OPENAI_SECOND[@])
    else
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo -e "${RED}ANTHROPIC_API_KEY not set. Please add it to $KEY_FILE or export it.${NC}"
            exit 1
        fi
        selected_validator_model=$(choose_api_model_inline "Anthropic" ANTHROPIC_HIGH[@] ANTHROPIC_SECOND[@])
    fi

    # Profiling
    echo ""
    echo -n "Enable performance profiling? [Y/n]: "
    read -r profile_choice
    profile_choice=${profile_choice:-y}
    if [[ "$profile_choice" =~ ^[Yy]$ ]]; then
        profile_flag="--profile"
    else
        profile_flag=""
    fi

    # Quiet mode
    echo -n "Quiet mode (minimal output)? [y/N]: "
    read -r quiet_choice
    quiet_choice=${quiet_choice:-n}
    if [[ "$quiet_choice" =~ ^[Yy]$ ]]; then
        quiet_flag="-q"
    else
        quiet_flag=""
    fi

    if [ -z "$PYTHON_BIN" ]; then
        if ! detect_python; then
            echo -e "${RED}Python 3 executable not found.${NC}"
            exit 1
        fi
    fi

    # Build command
    cmd=("$PYTHON_BIN" "sentencemaker.py" "-w" "$wordlist" "-o" "$output" "--max-words" "$max_words" "--min-words" "$min_words" "--max-sentences" "$max_sentences" "--gen-llm-provider" "$gen_provider" "--gen-llm-model" "$selected_model" "--llm-provider" "$val_provider" "--llm-model" "$selected_validator_model" "--validator-llm-model" "$selected_validator_model")
    if [ -n "$profile_flag" ]; then
        cmd+=("$profile_flag")
    fi
    if [ -n "$quiet_flag" ]; then
        cmd+=("$quiet_flag")
    fi

    # Show summary
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Configuration Summary${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo "  Word list: $wordlist"
    echo "  Output: $output"
    echo "  Max words/sentence: $max_words"
    if [ "$max_sentences" -gt 0 ]; then
        echo "  Max sentences: $max_sentences"
    else
        echo "  Max sentences: until all words covered"
    fi
    describe_provider() {
        if [[ "$1" == "ollama" ]]; then
            echo "LOCAL (FREE via Ollama)"
        elif [[ "$1" == "openai" ]]; then
            echo "OpenAI API (costs $)"
        elif [[ "$1" == "anthropic" ]]; then
            echo "Anthropic API (costs $)"
        else
            echo "$1"
        fi
    }
    echo "  LLM provider (generation): $(describe_provider "$gen_provider")"
    echo "  LLM model (generation): ${selected_model}"
    echo "  LLM provider (validation): $(describe_provider "$val_provider")"
    echo "  LLM model (validation): ${selected_validator_model}"
    echo "  Profiling: $([ -n "$profile_flag" ] && echo 'Yes' || echo 'No')"
    echo "  Quiet mode: $([ -n "$quiet_flag" ] && echo 'Yes' || echo 'No')"
    echo ""
    printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -n "Run with these settings? [Y/n]: "
    read -r confirm
    confirm=${confirm:-y}

    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo ""
        printf '%q ' "${cmd[@]}" | sed 's/ $//' > "$LAST_RUN_FILE"
        "${cmd[@]}"
    else
        echo "Cancelled."
        exit 0
    fi
}

# Main execution
main() {
    # Step 1: Cleanup
    cleanup_processes

    # Step 2: Start ollama
    start_ollama || {
        echo -e "\n${RED}Cannot continue: Ollama must be running for SentenceMaker.${NC}"
        echo "Please start it manually (e.g., 'ollama serve') and re-run ./start.sh"
        exit 1
    }

    # Step 3: Check environment
    check_environment || {
        echo -e "\n${RED}Please activate the sentencemaker environment first:${NC}"
        echo "  conda activate sentencemaker"
        exit 1
    }

    # Step 4: Check models
    check_models

    # Detect unfinished run
    STATE_FILES=($(detect_checkpoint_files))
    if [ -n "${STATE_FILES[*]}" ]; then
        resume_previous_run "${STATE_FILES[@]}" || true
    fi

    # Run configuration wizard
    run_wizard
}

# Run main function
main
