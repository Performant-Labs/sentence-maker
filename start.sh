#!/bin/bash
# SentenceMaker Startup Script
# Ensures clean environment and starts the sentence generator

set -e  # Exit on error

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
        if curl -s --max-time 10 http://localhost:11434/api/generate -d '{"model": "gemma2:9b", "prompt": "Test", "stream": false, "options": {"num_predict": 5}}' 2>&1 | grep -q "response"; then
            local gen_end=$(date +%s.%N)
            local gen_time=$(echo "$gen_end - $gen_start" | bc)
            echo -e "${GREEN}✓ Working (${gen_time}s)${NC}"
            
            # Warn if slow
            if (( $(echo "$gen_time > 5" | bc -l) )); then
                echo -e "  ${YELLOW}⚠ Generation is slow (>5s for 5 tokens)${NC}"
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
    done < <(find "$PWD" -maxdepth 5 -name '*.state.json' ! -path "./venv/*" -print0)
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
    local test_response=$(curl -s --max-time 10 "${OLLAMA_URL}/api/generate" -d '{
        "model": "gemma2:9b",
        "prompt": "Test",
        "stream": false,
        "options": {"num_predict": 5}
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
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "  Ollama PID: $OLLAMA_PID"
    
    # Wait for ollama to be ready (up to 15 seconds with progress)
    for i in {1..15}; do
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
        echo "    ollama pull gemma2:9b   # Recommended default (best balance)"
        echo "    ollama pull gemma2:27b  # Highest quality (slow)"
        echo "    ollama pull mistral:7b-instruct-v0.2-q4_0  # Faster alternative"
        return 0
    fi
    
    echo "  Available models:"
    # Highlight recommended models
    while IFS= read -r model; do
        if [[ "$model" == "gemma2:9b" ]]; then
            echo -e "    • $model ${GREEN}(recommended default)${NC}"
        elif [[ "$model" == "gemma2:27b" ]]; then
            echo -e "    • $model ${YELLOW}(highest quality, slower)${NC}"
        elif [[ "$model" == "mistral:7b-instruct"* ]]; then
            echo -e "    • $model ${GREEN}(recommended - faster)${NC}"
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
    echo -n "Word list file [words/words.txt]: "
    read -r wordlist
    wordlist=${wordlist:-words/words.txt}
    
    # Output file
    echo -n "Output file [output/sentences.txt]: "
    read -r output
    output=${output:-output/sentences.txt}
    
    # Max words
    echo -n "Maximum words per sentence [15]: "
    read -r max_words
    max_words=${max_words:-15}

    echo -n "Maximum sentences to generate (0 = until done) [0]: "
    read -r max_sentences
    max_sentences=${max_sentences:-0}
    
    echo ""
    echo -e "${YELLOW}LLM Model Selection:${NC}"
    echo "LLM generates semantically coherent sentences (required)."
    echo ""
    
    # Check if Ollama is responsive
    if ! check_ollama; then
        echo ""
        echo -e "${RED}Cannot continue without a responsive Ollama server.${NC}"
        exit 1
    fi
    
    selected_model="gemma2:9b"
        
    echo ""
    # Show available models
    echo "Available LLM models:"
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || true)
    if [ -n "$MODELS" ]; then
        i=1
        default_num=""
        declare -a model_array
        while IFS= read -r model; do
            model_array[$i]=$model
            if [[ "$model" == "gemma2:9b" ]]; then
                echo -e "  $i. $model ${GREEN}(recommended - best balance)${NC}"
                default_num=$i
            elif [[ "$model" == "gemma2:27b" ]]; then
                echo -e "  $i. $model ${YELLOW}(highest quality, slower)${NC}"
            elif [[ "$model" == "mistral:7b-instruct"* ]]; then
                echo -e "  $i. $model ${YELLOW}(faster, good quality)${NC}"
            else
                echo "  $i. $model"
            fi
            ((i++))
        done <<< "$MODELS"
        
        echo ""
        if [ -n "$default_num" ]; then
            echo -n "Select model number [$default_num for ${model_array[$default_num]}]: "
            read -r model_num
            model_num=${model_num:-$default_num}
        else
            echo -n "Select model number [1]: "
            read -r model_num
            model_num=${model_num:-1}
        fi
        
        if [ -n "${model_array[$model_num]}" ]; then
            selected_model=${model_array[$model_num]}
        fi
    else
        echo -e "${YELLOW}No models found. Using default: ${selected_model}${NC}"
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
    cmd=("$PYTHON_BIN" "sentencemaker.py" "-w" "$wordlist" "-o" "$output" "--max-words" "$max_words" "--max-sentences" "$max_sentences" "--llm-model" "$selected_model")
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
    echo "  LLM model: ${selected_model}"
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
