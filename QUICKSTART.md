# SentenceMaker Quick Start

## 1. One-Command Startup

```bash
./start.sh
```

The wizard will:
1. Clean up any hung Python/Ollama processes.
2. Start (or restart) the Ollama server.
3. Activate the `sentencemaker` conda env.
4. Confirm recommended models (gemma2:9b by default).
5. Prompt for word list, output path, max words per sentence, **max sentences (0 = until all words)**, LLM model, profiling, and quiet mode.

If a checkpoint (`*.state.json`) exists, you’ll be offered a **Resume? [Y/n]** prompt before the wizard begins. Accepting replays the last command automatically; declining deletes the checkpoint and starts fresh.

## 2. Watch Progress

Terminal 1 (wizard) shows live status and rejection counts. Terminal 2 (optional) can display the stats snapshot generated after every sentence:

```bash
watch -n 2 cat output/sentences.txt.stats.txt
```

The snapshot lists sentences generated, coverage, rejection counts, template failures, and the sentence-duration histogram.

To inspect the output itself:

```bash
tail -f output/sentences.txt
```

## 3. Run Manually (without wizard)

```bash
python sentencemaker.py \
  -w words/words.txt \
  -o output/sentences.txt \
  --max-words 15 \
  --max-sentences 500 \
  --llm-model gemma2:9b
```

Key flags:

| Flag | Purpose | Default |
|------|---------|---------|
| `--max-words` | Max words per sentence | `15` |
| `--max-sentences` | Stop after N sentences (`0` = use all words) | `0` |
| `--llm-model` | Ollama model (`gemma2:9b`, `gemma2:27b`, `mistral:7b-…`) | `gemma2:9b` |
| `--profile` | Print timing breakdown | off |
| `-q` | Quiet mode (stats file still updates) | off |

## 4. Stopping and Resuming

- Press `Ctrl+C` to stop safely; the generator finishes the current sentence, saves the checkpoint, and exits.
- Next run, `./start.sh` detects the checkpoint and offers to resume; choose `Y` to continue where you left off or `N` to delete the checkpoint and start over.

## 5. Tips

- **Fewer sentences?** Set `--max-sentences N`.
- **Higher quality?** Use `--llm-model gemma2:27b` (requires more VRAM/time).
- **Faster runs?** Try `mistral:7b-instruct-v0.2-q4_0`.
- **Monitor semantics?** Check `output/sentences.txt.stats.txt` for heuristic vs. validator rejections and top failing templates.
- **Trouble with Ollama?** Run `./start.sh --reset` to restart the server automatically.

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| “Error: This program requires Python 3.8-3.12” | `conda activate sentencemaker` |
| “Cannot connect to Ollama” | `ollama serve &` (or let `./start.sh` restart it) |
| Slow generation (>5s/sentence) | `./start.sh --reset`, close other apps, or switch to a smaller model |
| Stats file missing | Wait until the run starts; the `.stats.txt` file is created after the first sentence |

You’re ready—run `./start.sh`, choose your settings, and watch the stats file for live feedback.***
