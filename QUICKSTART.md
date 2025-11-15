# SentenceMaker Quick Start Guide

## One-Command Startup

```bash
./start.sh
```

This will:
1. ✓ Kill any hung processes (gentle → forceful)
2. ✓ Start ollama server (if not running)
3. ✓ Activate conda environment
4. ✓ Check available models
5. ✓ Show run options menu

## Run Options

### 1. Quick Run (Fast, No Filtering)
**Best for:** Testing, quick results
```bash
python sentencemaker.py
```
- Time: ~10-15 minutes
- Output: ~1500 sentences
- Coverage: 95%
- Quality: Some nonsense

### 2. Standard Run (Balanced)
**Best for:** General use
```bash
python sentencemaker.py --max-words 15
```
- Time: ~12-18 minutes
- Output: ~1200 sentences
- Coverage: 94%
- Quality: Some nonsense

### 3. Minimized Output (Fewer Sentences)
**Best for:** Compact output
```bash
python sentencemaker.py --max-words 18 --minimize
```
- Time: ~15-20 minutes
- Output: ~800 sentences
- Coverage: 92%
- Quality: Some nonsense

### 4. With LLM Filtering (Sensible Sentences Only)
**Best for:** Quality over quantity
```bash
python sentencemaker.py --max-words 15 --llm-filter
```
- Time: ~15-20 minutes
- Output: ~850 sentences (70% kept)
- Coverage: 90%
- Quality: All sensible ✓

### 5. Full Featured (All Options) ⭐ RECOMMENDED
**Best for:** Best results with detailed stats
```bash
python sentencemaker.py --max-words 18 --minimize --llm-filter --profile
```
- Time: ~20-25 minutes
- Output: ~600 sentences (70% kept)
- Coverage: 88%
- Quality: All sensible ✓
- Profiling: Detailed performance breakdown

### 6. Custom Word List
**Best for:** Your own vocabulary
```bash
python sentencemaker.py -w custom_words.txt -o output/custom.txt --max-words 12
```

### 7. Quiet Mode with LLM
**Best for:** Background processing
```bash
python sentencemaker.py --max-words 15 --llm-filter -q
```

## Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-w, --wordlist FILE` | Input word list | `words/words.txt` |
| `-o, --output FILE` | Output file | `output/sentences.txt` |
| `--max-words N` | Max words per sentence | 10 |
| `--minimize` | Optimize for fewer sentences | Off |
| `--llm-filter` | Filter with local LLM | Off |
| `--llm-model MODEL` | Ollama model to use | `deepseek-r1:1.5b` |
| `--profile` | Show performance breakdown | Off |
| `-q, --quiet` | Suppress progress output | Off |

## Watch Progress Live

Open a second terminal:

```bash
# Watch sentences being generated
tail -f output/sentences.txt

# Count sentences
watch -n 1 'wc -l output/sentences.txt'

# See last 10 sentences
watch -n 1 'tail -10 output/sentences.txt'
```

## Troubleshooting

### "Error: This program requires Python 3.8-3.12"
```bash
conda activate sentencemaker
```

### "Ollama not running"
```bash
ollama serve &
```

### "No models found"
```bash
ollama pull deepseek-r1:1.5b
```

### Hung process
```bash
./start.sh  # Will clean up automatically
```

## Performance Comparison

| Configuration | Time | Sentences | Coverage | Quality |
|---------------|------|-----------|----------|---------|
| Quick | 10 min | 1500 | 95% | Mixed |
| Standard | 15 min | 1200 | 94% | Mixed |
| Minimized | 18 min | 800 | 92% | Mixed |
| + LLM Filter | 20 min | 850 | 90% | ✓ Good |
| Full Featured | 25 min | 600 | 88% | ✓ Good |

## Recommended Models

| Model | Speed | Quality | RAM | Command |
|-------|-------|---------|-----|---------|
| **gemma2:27b** ⭐ | ⚡⚡ | ⭐⭐⭐⭐⭐ | 15GB | `ollama pull gemma2:27b` (DEFAULT) |
| mistral:7b-instruct | ⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB | `ollama pull mistral:7b-instruct-v0.2-q4_0` |
| deepseek-r1:8b | ⚡⚡⚡ | ⭐⭐⭐ | 5GB | `ollama pull deepseek-r1:8b` |

## Example Workflow

```bash
# 1. Start everything
./start.sh

# 2. Choose option 5 (Full Featured)

# 3. In another terminal, watch progress
tail -f output/sentences.txt

# 4. Wait for completion (~25 minutes)

# 5. Review output
head -20 output/sentences.txt
wc -l output/sentences.txt
```

## Tips

- **First run?** Use option 2 (Standard) to test
- **Want quality?** Use option 4 or 5 with LLM filtering
- **In a hurry?** Use option 1 (Quick)
- **Large word list?** Use `--minimize` to reduce output
- **Debugging?** Add `--profile` to see performance breakdown
- **Background job?** Add `-q` for quiet mode

## Support

For issues or questions, check:
- README.md - Full documentation
- setup.sh - Environment setup
- start.sh - This startup script
