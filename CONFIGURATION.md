# SentenceMaker Configuration Guide

All configurable parameters are now organized at the top of `sentence_generator.py` in clearly documented sections.

## Quick Reference

### Key Parameters You May Want to Adjust

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `LLM_WORD_POOL_SIZE` | Line 37 | `100` | Words per POS given to LLM (50-100 recommended) |
| `LLM_MAX_ATTEMPTS` | Line 45 | `5` | Retry attempts before fallback (2-10 recommended) |
| `SIMILARITY_THRESHOLD` | Line 133 | `0.55` | Reject sentences above this similarity score |
| `MAX_SIMILARITY_HISTORY` | Line 132 | `400` | Number of recent sentences to compare against |
| `MAX_START_MEMORY` | Line 123 | `60` | Recent sentence starts to track for variety |

## Configuration Sections

### 1. Generation Parameters (Lines 25-45)

**`LLM_WORD_POOL_SIZE = 100`**
- Controls how many words per POS category the LLM receives
- **Impact**: Higher = more semantic options, better coherence
- **Trade-off**: Larger prompts, slightly slower generation (~0.1-0.2s per sentence)
- **Recommended range**: 50-100 words
- **History**: Originally 20 → 75 → 100 for maximum quality

**`LLM_MAX_ATTEMPTS = 5`**
- Number of times to retry LLM generation before falling back to simple sentences
- **Impact**: Higher = fewer fallback sentences, better quality
- **Trade-off**: More retries = slower generation when validation fails
- **Recommended range**: 3-10 attempts
- **History**: Originally 2 → 5 to complement stricter validation

### 2. Semantic Validation (Lines 38-68)

**Color Word Validation**
- `COLOR_WORDS`: Spanish color words to validate
- `COLOR_CONTEXT_WORDS`: Physical objects that can have colors
- `ABSTRACT_COLOR_WORDS`: Abstract concepts that should NOT have colors
- **Purpose**: Prevents nonsensical combinations like "red emotion" or "blue time"

### 3. Question Words (Lines 69-78)

**`QUESTION_WORDS`**
- Spanish interrogative words (qué, quién, dónde, etc.)
- Used to detect when templates should generate questions

### 4. Topic Seeding (Lines 79-99)

**`SEED_TOPICS`**
- 12 topic domains with associated keywords
- When ≥2 keywords match available words, guides LLM to that topic
- **Topics**: daily routine, family life, work, home, food, nature, education, emotions, travel, health, leisure, shopping
- **Purpose**: Improves semantic coherence by giving contextual guidance

### 5. Performance & Statistics (Lines 100-108)

**`DURATION_BUCKETS`**
- Time buckets for generation histogram: `["<4s", "4-8s", "8-12s", "12-16s", "16-20s", ">=20s"]`

**`WORD_PATTERN`**
- Regex for extracting Spanish words (includes accented characters)

### 6. Repetition Detection (Lines 110-124)

**`REPETITION_STOPWORDS`**
- Common function words to ignore when checking repetition
- Includes articles, prepositions, conjunctions (de, la, el, y, etc.)

**Sentence Start Tracking**
- `MAX_START_MEMORY = 60`: Track last 60 sentence starts
- `START_REPEAT_WINDOW = 50`: Check for duplicates in last 50
- **Purpose**: Prevents sentences from starting the same way repeatedly

### 7. Similarity Detection (Lines 126-138)

**Multi-dimensional similarity scoring** to prevent repetitive sentences:

- `MAX_SIMILARITY_HISTORY = 400`: Compare against last 400 sentences
- `SIMILARITY_THRESHOLD = 0.55`: Reject if similarity exceeds 55%
- `SIMILARITY_THRESHOLD_FLOOR = 0.25`: Minimum threshold (adaptive)
- `SIMILARITY_ADAPT_MIN_SCORES = 80`: Scores needed before adapting
- `SIMILARITY_ADAPT_MARGIN = 0.02`: Threshold adjustment increment
- `SIMILARITY_WEIGHTS = (0.4, 0.2, 0.4)`: Weights for (lexical, structural, semantic)
- `SENTENCE_EMBEDDING_MODEL`: Model for semantic similarity

## How to Modify

### Example: Increase Word Pool for Better Quality

```python
# In sentence_generator.py, line 36
LLM_WORD_POOL_SIZE = 100  # Increased from 75
```

**Expected result**: More semantic variety, potentially better sentences, slightly slower

### Example: Reduce Similarity Threshold for More Variety

```python
# In sentence_generator.py, line 133
SIMILARITY_THRESHOLD = 0.45  # Reduced from 0.55
```

**Expected result**: More diverse sentences, but may accept some similar ones

### Example: Add New Topic Domain

```python
# In sentence_generator.py, add to SEED_TOPICS dict
"technology": ["computadora", "internet", "teléfono", "aplicación", "programa", "digital"],
```

**Expected result**: LLM will generate tech-themed sentences when these words are available

## Testing Configuration Changes

After modifying any configuration:

1. **Test with small run**:
   ```bash
   conda run -n sentencemaker python sentencemaker.py --max-sentences 10 --quiet
   ```

2. **Check output quality**:
   ```bash
   cat output/sentences.txt
   ```

3. **Review statistics**:
   ```bash
   cat output/sentences.txt.stats.txt
   ```

4. **Monitor rejection rates** - Higher rejections = stricter validation (good for quality)

## Configuration Best Practices

1. **Start conservative**: Use default values first
2. **Change one parameter at a time**: Easier to identify impact
3. **Test with small runs**: 10-50 sentences before full generation
4. **Monitor rejection rates**: 20-40% is healthy; >60% may be too strict
5. **Balance quality vs speed**: Higher word pools = better quality but slower
