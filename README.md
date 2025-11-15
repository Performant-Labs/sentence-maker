# SentenceMaker

A Python application that generates semantically coherent, grammatically correct Spanish sentences from a word list using local LLM (Large Language Model) generation. The program uses NLP to classify words and leverages LLM intelligence to create natural, meaningful sentences while ensuring comprehensive vocabulary coverage.

## Features

- **LLM-Powered Generation**: Uses local Ollama LLM (gemma2:27b) to generate semantically coherent sentences
- **Smart Word Classification**: Uses spaCy to automatically classify Spanish words by part of speech
- **Thematic Variety**: 12 seed topics (family, work, daily routine, etc.) guide sentence generation for natural contexts
- **Template-Guided Structure**: 54 grammatical templates ensure diverse sentence structures (questions, commands, subjunctive, etc.)
- **High Coverage**: Typically achieves 85-90% vocabulary coverage
- **Semantic Coherence**: LLM ensures sentences make logical sense, not just grammatical correctness
- **Multiple Sentence Types**: Questions, commands, declarative statements, complex sentences with connectors
- **All Three Moods**: Indicative, Imperative, and Subjunctive with proper usage
- **Configurable**: Customize maximum sentence length and LLM model
- **Real-Time Progress**: Live timer and sentence count during generation
- **Incremental Saving**: Output file updated after each sentence
- **Interactive Wizard**: User-friendly start script guides configuration
- **Ollama Health Check**: Automatic detection and restart of stuck LLM server

## Installation

**Requirements**: 
- Python 3.8-3.12 (Python 3.13 not yet supported due to spaCy/blis compatibility)
- **Ollama** with gemma2:27b model (for LLM generation)

### Step 1: Install Ollama

Install Ollama and download the recommended model:

```bash
# Install Ollama (macOS/Linux)
curl https://ollama.ai/install.sh | sh

# Download recommended model (15GB)
ollama pull gemma2:27b

# Start Ollama server
ollama serve &
```

For other operating systems, visit: https://ollama.ai

### Step 2: Install Python Dependencies

**Important**: Choose **ONE** method below (conda OR venv, never both).

#### Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup.sh
```

This will:
- Create a conda environment with Python 3.12
- Install all dependencies
- Download the spaCy Spanish model

Then activate:
```bash
conda activate sentencemaker
```

### Manual Setup with Conda

**Use this if you have conda/miniconda installed:**

```bash
# Deactivate any active venv first
deactivate  # (if in venv)

# Create environment with Python 3.12
conda create -n sentencemaker python=3.12 -y

# Activate environment
conda activate sentencemaker

# Install dependencies
pip install -r requirements.txt

# Download Spanish language model
python -m spacy download es_core_news_sm
```

### Using venv (Alternative)

**Only use this if:**
- You don't have conda installed, AND
- Your system Python is 3.8-3.12 (check with `python3 --version`)

```bash
# Verify Python version first
python3 --version  # Must be 3.8-3.12, NOT 3.13

# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Spanish language model
python -m spacy download es_core_news_sm
```

**Note**: If your system has Python 3.13, venv won't work. Use conda instead.

### Troubleshooting Installation

**Problem: "Error: This program requires Python 3.8-3.12"**
- Your Python version is incompatible
- Solution: Use conda to create Python 3.12 environment (see above)

**Problem: Mixed environments (both venv and conda active)**
```bash
# Your prompt shows both: (venv) (base)
# Solution: Deactivate venv first
deactivate
conda activate sentencemaker
```

**Problem: "blis" compilation errors**
- You're using Python 3.13
- Solution: Use conda with Python 3.12 (see Quick Setup above)

## Usage

### Quick Start (Recommended)

Use the interactive start script:

```bash
./start.sh
```

The wizard will guide you through:
1. Word list selection (default: `words/words.txt`)
2. Output file location (default: `output/sentences.txt`)
3. Maximum words per sentence (default: 15)
4. LLM model selection (shows available models)
5. Configuration confirmation

**With Ollama Reset** (if LLM is stuck):
```bash
./start.sh --reset
```

This will:
- Gracefully stop Ollama (5 second timeout)
- Force kill if needed
- Restart Ollama server
- Run the configuration wizard

### Direct Usage

Run directly with command-line arguments:

```bash
# Basic usage with LLM
python sentencemaker.py --llm-filter

# Custom configuration
python sentencemaker.py -w words/words.txt -o output/sentences.txt --max-words 15 --llm-filter

# Use different LLM model
python sentencemaker.py --llm-filter --llm-model mistral:7b-instruct-v0.2-q4_0
```

**Note**: `--llm-filter` is required for generation. The program uses LLM to create sentences.

### Command-Line Arguments

```bash
# Required
--llm-filter              # Enable LLM generation (required)

# Optional
-w, --wordlist PATH       # Word list file (default: words/words.txt)
-o, --output PATH         # Output file (default: output/sentences.txt)
--max-words N             # Max words per sentence (default: 15)
--llm-provider PROVIDER   # LLM provider: ollama (default), openai, anthropic
--llm-model MODEL         # LLM model (default: gemma2:9b)
-q, --quiet               # Suppress progress output
--profile                 # Enable performance profiling
```

### Examples

```bash
# Standard usage
python sentencemaker.py --llm-filter

# Custom configuration
python sentencemaker.py -w my_words.txt -o my_output.txt --max-words 12 --llm-filter

# Use faster model
python sentencemaker.py --llm-filter --llm-model mistral:7b-instruct-v0.2-q4_0

# Quiet mode with profiling
python sentencemaker.py --llm-filter -q --profile
```

### Live Monitoring

The program provides real-time progress updates:

**Progress Display:**
```
Generating sentences (max 15 words per sentence)...
Total words to cover: 10075
Available templates: 54

Starting generation...
  Generating sentence 1... 2.3s elapsed (calling LLM...)
  Generating sentence 2... 4.6s elapsed (calling LLM...)
  Generated 50 sentences, 9523 words remaining (115.0s)
  Generating sentence 51... 117.3s elapsed (calling LLM...)
  Generated 50 sentences, 9882 words remaining (22.9s) | rejections: 51 (heuristic 36, validator 15)
  Generating sentence 66... 82.1s elapsed (calling LLM...) | rejections: 67 (heuristic 49, validator 18)
```

- `Generated …` lines print every 50 sentences. The time in parentheses is the batch duration. The rejection summary splits discards into **heuristic** rejections (our local semantic filters) and **validator** rejections (LLM coherence check). Heuristic failures are caught before we spend an LLM call, whereas validator failures happen when the LLM still finds grammatical/semantic issues.
- `Generating sentence …` lines show the current LLM call. The timer is how long that attempt has been running, and the same cumulative rejection breakdown appears at the end so you can see how strict each stage has been during the run.
- A live stats snapshot is written after every sentence/rejection to `<output>.stats.txt` (for the default run: `output/sentences.txt.stats.txt`). Inspect it whenever you want a snapshot—for example:
  ```bash
  watch -n 2 cat output/sentences.txt.stats.txt
  ```

**Watch Output File in Real-Time:**

Open a second terminal to monitor the output file as it's being written:

```bash
# Watch file grow in real-time
tail -f output/sentences.txt

# Count lines as they're added
watch -n 1 'wc -l output/sentences.txt'

# See last 10 sentences
watch -n 1 'tail -10 output/sentences.txt'
```

The output file is:
- **Cleared at start** - Fresh file for each run
- **Updated after each sentence** - Real-time incremental saves
- **Safe from crashes** - Never lose progress

### Example Word List Format

Create a text file with one Spanish word per line:
```
perro
gato
casa
grande
correr
comer
rápido
feliz
```

### Unused Words File

If not all words can be used in grammatically correct sentences, the program automatically creates a `*_unused.txt` file showing:
- Which words weren't used
- Their part of speech (POS)
- Their morphological features (gender, number, mood)
- Why they might not have been usable

Example `output/sentences_unused.txt`:
```
# Unused Words (2500 total)

## VERB (1200 words)
correr (gender=-, number=-, mood=-)
estudiar (gender=-, number=-, mood=-)
nadar (gender=-, number=-, mood=-)

## NOUN (800 words)
libertad (gender=Fem, number=Sing, mood=-)
esperanza (gender=Fem, number=Sing, mood=-)

## ADJ (500 words)
increíble (gender=-, number=-, mood=-)
```

This helps you understand which words couldn't be incorporated and why.

**Note**: With automatic verb conjugation enabled, infinitives like "correr", "estudiar" are automatically conjugated (e.g., "corre", "estudia") and marked as used. The unused file will primarily contain words with missing morphological features or incompatible POS combinations.

### Performance Profiling

Use the `--profile` flag to see detailed performance breakdown:

```bash
python sentencemaker.py --profile
```

Example output:
```
PERFORMANCE BREAKDOWN (Python vs Compiled)
------------------------------------------------------------
Python code time:        1.234s (15.2%)
Compiled libraries:      6.890s (84.8%)
  ├─ spaCy (NLP):        6.500s
  └─ mlconjug3 (verbs):  0.390s
------------------------------------------------------------
Profiled time total:     8.124s

Note: Compiled libraries (C/Cython) are 10-100x faster than Python
```

This shows:
- **Python code time**: Your custom logic (sentence generation, template matching)
- **Compiled libraries**: Fast C/Cython code (spaCy NLP, verb conjugation)
- **spaCy**: Most time spent in NLP analysis (word classification, morphology)
- **mlconjug3**: Time spent conjugating verbs

**Key insight**: ~85% of execution time is in highly-optimized compiled code, which is why the program is fast despite Python being "slow."

## How It Works

### 1. Word Classification
Uses spaCy's Spanish NLP model to classify each word by part of speech (noun, verb, adjective, etc.)

### 2. Template Selection
Chooses from 54 grammatical templates to guide sentence structure:
- Simple declarative sentences (indicative mood)
- Questions (¿Qué?, ¿Dónde?, ¿Cuándo?, ¿Cómo?, ¿Quién?, ¿Por qué?)
- Complex sentences with subordinate clauses (que, porque, cuando, si, aunque)
- Compound sentences (y, pero, o)
- Imperative commands (¡Come!, ¡No corras!)
- Subjunctive mood (Espero que..., Quiero que..., Ojalá que...)

Templates provide **structural variety** - they tell the LLM what type of sentence to create.

### 3. Seed Topic Selection
Intelligently selects a thematic context based on available words:

**12 Seed Topics:**
- Daily routine, Family life, Work, Home, Food, Nature
- Education, Emotions, Travel, Health, Leisure, Shopping

**Smart Selection:**
- Scores topics by keyword matches in available words
- Uses best matching topic if 2+ words match
- Falls back to random topic or no topic
- Example: ["madre", "hijo", "familia"] → "family life" seed

### 4. LLM Generation
Sends a structured prompt to the local LLM (gemma2:27b via Ollama):

```
Create a declarative statement about family life in Spanish using ONLY words from this list:
NOUN: madre, hijo, casa, familia
VERB: come, trabaja, vive

Requirements:
- Use MOST of the words (may skip 1-2 if they don't fit naturally)
- Prioritize semantic coherence and natural meaning
- Make it sound like a native speaker would say
- Keep it under 15 words
- Return ONLY the sentence
```

**LLM generates:** "La madre trabaja mientras el hijo come en casa con la familia."

### 5. Word Tracking
- Extracts which words were actually used in the sentence
- Marks those words as used (removes from unused pool)
- Continues until all words are covered or max attempts reached
- Typically achieves 85-90% coverage

### Why LLM Instead of Templates?

**Old approach (template-based):**
- Fill template slots with random words
- Result: Grammatically correct but often nonsensical
- Example: "El músico hacía trabajar las tablas mediante la luz." ❌

**New approach (LLM-guided):**
- LLM generates semantically coherent sentences
- Templates guide structure, LLM ensures meaning
- Example: "El músico trabaja con las tablas en el taller." ✅

## Project Structure

```
SentenceMaker/
├── start.sh                 # Interactive startup script with Ollama health check
├── sentencemaker.py         # Main application entry point
├── sentence_generator.py    # LLM-based sentence generation
├── llm_validator.py         # LLM interface (Ollama/OpenAI/Anthropic)
├── word_classifier.py       # Word classification with spaCy
├── sentence_templates.py    # 54 Spanish grammar templates
├── requirements.txt         # Python dependencies
├── setup.sh                 # Automated environment setup
├── words/                   # Input folder for word lists
│   └── words.txt           # Your Spanish word list (place here)
├── output/                  # Output folder for generated sentences
│   ├── sentences.txt       # Generated sentences (created automatically)
│   └── sentences_unused.txt # Unused words (if coverage < 100%)
├── README.md               # This file
└── QUICKSTART.md           # Quick reference guide
```

## Performance

On a MacBook M1 Pro with 24GB RAM:

**With LLM (gemma2:27b):**
- **10,000 words**: ~35-45 minutes total
- **Setup time**: 3-4 seconds (loading spaCy model)
- **LLM generation**: 2-3 seconds per sentence
- **Sentences generated**: ~800-1000
- **Memory usage**: ~500MB (Python) + ~8GB (Ollama/LLM)
- **Coverage**: 85-90% of words

**Performance Tips:**
- Use `./start.sh --reset` if generation is slow (>5s per sentence)
- Faster model: `mistral:7b-instruct-v0.2-q4_0` (~1.5s per sentence)
- Ensure Ollama server is responsive before starting
- Close other memory-intensive applications

**Speed Comparison:**
| Model | Size | Speed/Sentence | Quality | Total Time (10K words) |
|-------|------|----------------|---------|------------------------|
| gemma2:27b | 15GB | 2-3s | ⭐⭐⭐⭐⭐ | 35-45 min |
| mistral:7b | 4GB | 1.5-2s | ⭐⭐⭐⭐ | 20-30 min |

## Example Output

**LLM-generated sentences (semantically coherent):**

```
Cuando aprueba las tareas, podrás vivir en zonas turísticas y ocupar un molino.
Si la familia conserva fortaleza, esperamos cooperación pero carecen de retroalimentación.
Traté de comprarlo durante la jornada.
Dejó la creatividad al pasar hacia los desarrollos.
Cuando salga el sol, aprovecharé la mañana y dedicarse a la fotografía.
Las emociones reflejan la sensibilidad humana según las circunstancias.
El perro intenta alcanzar el piano, pero se encuentra sin acceso.
Cuando llega el plazo, entregamos las obras y invitamos a los vecinos.
Evitar el consumo de alcohol mantiene las respectivas épocas saludables.
¿Dónde trabajan los arquitectos en la manufactura?
¡Trabaja como trabajador responsable!
Espero que los médicos demuestren precisión cuando trabajan.
```

**Note:** Sentences are grammatically correct and semantically coherent, though some may sound unusual due to forced word usage from the vocabulary list.

## Statistics

The program provides detailed statistics:
- Total words processed
- Coverage percentage
- Number of sentences generated
- Average words per sentence
- Word repetition metrics
- **Timing breakdown** for each stage:
  - Load word list
  - Initialize spaCy model
  - Classify words
  - Initialize generator
  - Generate sentences
  - Save output
  - Total execution time

## Requirements

- Python 3.8+
- spaCy 3.7+
- Spanish language model (`es_core_news_sm`)

## Quality Assurance

The LLM ensures:
- ✅ **Grammatically correct Spanish** - Proper conjugation, agreement, syntax
- ✅ **Semantic coherence** - Sentences make logical sense
- ✅ **Natural phrasing** - Sounds like a native speaker
- ✅ **Diverse structures** - Questions, commands, complex sentences
- ✅ **All three moods** - Indicative, Imperative, Subjunctive
- ✅ **Thematic variety** - 12 seed topics for realistic contexts

### Sentence Types Generated

**Declarative (Indicative):**
- "Las emociones reflejan la sensibilidad humana según las circunstancias."
- "Traté de comprarlo durante la jornada."

**Questions:**
- "¿Dónde trabajan los arquitectos en la manufactura?"

**Commands (Imperative):**
- "¡Trabaja como trabajador responsable!"

**Subjunctive:**
- "Espero que los médicos demuestren precisión cuando trabajan."

**Complex sentences:**
- "Cuando llega el plazo, entregamos las obras y invitamos a los vecinos."
- "Si la familia conserva fortaleza, esperamos cooperación pero carecen de retroalimentación."

## Word frequency source

https://wortschatz.uni-leipzig.de/en/download/spa#web

## Limitations

- **Spanish language only** - Designed specifically for Spanish
- **Requires Ollama** - Local LLM server must be running (free, no API costs)
- **Memory intensive** - LLM requires ~8GB RAM for gemma2:27b model
- **Generation speed** - 2-3 seconds per sentence (~35-45 min for 10K words)
- **Word coverage** - Typically 85-90% (some words may not fit naturally)
- **Semantic quality** - Sentences are coherent but may sound unusual due to forced word combinations
- **Internet required** - For initial setup (spaCy model, Ollama installation)
- **System requirements** - Works best on systems with 16GB+ RAM

## License

MIT License
