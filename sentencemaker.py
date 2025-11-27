#!/usr/bin/env python3
"""
SentenceMaker - Spanish Sentence Generator
Generates grammatically correct Spanish sentences using a word list.
"""
import argparse
import time
import sys
import os
import cProfile
import pstats
from io import StringIO
from pathlib import Path

# Check Python version
if sys.version_info < (3, 8) or sys.version_info >= (3, 13):
    print("Error: This program requires Python 3.8-3.12")
    print(f"Current version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("\nPython 3.13 is not yet supported due to spaCy/blis compatibility issues.")
    print("Please use Python 3.12 or earlier.")
    print("\nTo create a compatible environment:")
    print("  conda create -n sentencemaker python=3.12 -y")
    print("  conda activate sentencemaker")
    sys.exit(1)

from word_classifier import WordClassifier
from sentence_generator import SentenceGenerator

DURATION_BUCKETS = SentenceGenerator.DURATION_BUCKETS
LEGACY_DURATION_BUCKET = ">=9s"
RUN_COUNTER_FILE = Path(".sentencemaker_run_counter")


def format_duration(seconds: float) -> str:
    """Return duration formatted as HHh MMm SS.SSSs or milliseconds."""
    if seconds is None:
        return "00h 00m 00.000s"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}h {minutes:02d}m {secs:06.3f}s"


def load_word_list(filepath: str) -> list[str]:
    """
    Load words from a file.
    
    Args:
        filepath: Path to the word list file
        
    Returns:
        List of words
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def load_next_run_id() -> int:
    """Return a monotonically increasing run identifier and persist it."""
    try:
        current = 0
        if RUN_COUNTER_FILE.exists():
            content = RUN_COUNTER_FILE.read_text(encoding="utf-8").strip()
            if content:
                current = int(content)
        next_id = current + 1
        RUN_COUNTER_FILE.write_text(str(next_id), encoding="utf-8")
        return next_id
    except Exception:
        # Fall back to timestamp-based ID if the counter file is unavailable.
        return int(time.time())


def save_sentences(sentences: list[str], output_file: str):
    """
    Save generated sentences to a file.
    
    Args:
        sentences: List of sentences to save
        output_file: Path to output file
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"\nSentences saved to: {output_file}")
    except Exception as e:
        print(f"Error saving sentences: {e}")
        sys.exit(1)


def analyze_profiling_data(profiler) -> dict:
    """
    Analyze profiling data to categorize time spent in Python vs compiled libraries.
    
    Args:
        profiler: cProfile.Profile object
        
    Returns:
        Dictionary with time breakdown
    """
    # Get profiling stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    
    # Categorize function calls
    compiled_time = 0.0
    python_time = 0.0
    spacy_time = 0.0
    mlconjug_time = 0.0
    
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func
        
        # Categorize by module
        if 'spacy' in filename.lower() or 'thinc' in filename.lower() or 'cymem' in filename.lower():
            spacy_time += tt
            compiled_time += tt
        elif 'mlconjug' in filename.lower():
            mlconjug_time += tt
            compiled_time += tt
        elif filename.endswith(('.so', '.pyd', '.dll')):  # C extensions
            compiled_time += tt
        elif 'site-packages' not in filename and filename != '~':
            # Our Python code
            python_time += tt
    
    total_time = compiled_time + python_time
    
    return {
        'total_time': total_time,
        'python_time': python_time,
        'compiled_time': compiled_time,
        'spacy_time': spacy_time,
        'mlconjug_time': mlconjug_time,
        'python_percent': (python_time / total_time * 100) if total_time > 0 else 0,
        'compiled_percent': (compiled_time / total_time * 100) if total_time > 0 else 0
    }


def save_unused_words(unused_words: set[str], classifier, output_file: str):
    """
    Save unused words to a file, organized by part of speech.
    
    Args:
        unused_words: Set of words that weren't used
        classifier: WordClassifier instance for POS info
        output_file: Path to output file
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Organize by POS
        from collections import defaultdict
        unused_by_pos = defaultdict(list)
        for word in sorted(unused_words):
            info = classifier.get_word_info(word)
            pos = info.get('pos', 'UNKNOWN')
            unused_by_pos[pos].append(word)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Unused Words ({len(unused_words)} total)\n\n")
            
            for pos in sorted(unused_by_pos.keys()):
                words = unused_by_pos[pos]
                f.write(f"## {pos} ({len(words)} words)\n")
                for word in words:
                    info = classifier.get_word_info(word)
                    gender = info.get('gender', '-')
                    number = info.get('number', '-')
                    mood = info.get('mood', '-')
                    f.write(f"{word}")
                    if gender != '-' or number != '-' or mood != '-':
                        f.write(f" (gender={gender}, number={number}, mood={mood})")
                    f.write("\n")
                f.write("\n")
        
        print(f"Unused words saved to: {output_file}")
    except Exception as e:
        print(f"Error saving unused words: {e}")
        sys.exit(1)


def run_generator(args, stage_times, run_id: int):
    """
    Core generator logic (can be profiled).
    
    Args:
        args: Parsed command-line arguments
        stage_times: Dictionary to store timing information
        
    Returns:
        Tuple of (sentences, stats, generator, classifier)
    """
    # Load word list
    stage_start = time.time()
    if not args.quiet:
        print(f"\nLoading word list from: {args.wordlist}")
    
    words = load_word_list(args.wordlist)
    stage_times['load_wordlist'] = time.time() - stage_start
    
    if not args.quiet:
        print(f"Loaded {len(words)} words ({format_duration(stage_times['load_wordlist'])})")
    
    if len(words) == 0:
        print("Error: Word list is empty")
        sys.exit(1)
    
    # Initialize classifier
    stage_start = time.time()
    if not args.quiet:
        print("\nInitializing Spanish language model...")
    
    try:
        classifier = WordClassifier()
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        sys.exit(1)
    
    stage_times['init_classifier'] = time.time() - stage_start
    if not args.quiet:
        print(f"Model loaded ({format_duration(stage_times['init_classifier'])})")
    
    # Classify words
    stage_start = time.time()
    classifier.classify_words(words)
    stage_times['classify_words'] = time.time() - stage_start
    
    if not args.quiet:
        print(f"Classification time: {format_duration(stage_times['classify_words'])}")
    
    # Initialize LLM client for generation (validation can be none)
    llm_validator = None
    from llm_validator import LLMValidator
    gen_provider = args.gen_llm_provider or args.llm_provider
    gen_model = args.gen_llm_model or args.llm_model

    if gen_provider != 'none':
        if not args.quiet:
            provider_info = {
                'ollama': f'LOCAL (FREE) - {gen_model}',
                'openai': f'OpenAI API (COSTS $) - {gen_model}',
                'anthropic': f'Anthropic API (COSTS $) - {gen_model}',
                'none': 'No validation (heuristics only)'
            }
            print(f"\nInitializing LLM client: {provider_info.get(gen_provider, gen_provider)}")
            if gen_provider != 'ollama':
                print(f"  ⚠️  WARNING: Using API provider - this will incur costs!")
                print(f"  Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        stage_start = time.time()
        try:
            llm_validator = LLMValidator(model=gen_model, provider=gen_provider)
            stage_times['init_llm'] = time.time() - stage_start
            
            if not args.quiet:
                print(f"LLM client ready ({format_duration(stage_times['init_llm'])})")
        except (ConnectionError, TimeoutError, ValueError) as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error initializing LLM client: {e}")
            sys.exit(1)
    
    # Initialize generator with output file for incremental saving
    stage_start = time.time()
    if not args.quiet:
        print(f"\nOutput will be saved to: {args.output}")
        if args.llm_provider == 'none':
            print("(Using LLM for generation; validation: none (heuristics only))")
        else:
            print("(Using LLM for generation and validation)")
    generator = SentenceGenerator(
        classifier,
        max_words=args.max_words,
        output_file=args.output,
        llm_validator=llm_validator,
        generation_model=args.gen_llm_model or args.validator_llm_model or args.llm_model,
        generation_provider=args.gen_llm_provider or args.llm_provider,
        min_words=args.min_words,
        skip_llm_validation=args.llm_provider == 'none',
        max_sentences=args.max_sentences,
        enable_spellcheck=args.spellcheck,
        run_id=run_id
    )
    
    resume_state = None
    checkpoint = generator.load_checkpoint()
    if checkpoint and args.resume:
        current_checksum = SentenceGenerator.compute_wordlist_checksum(words)
        stored_checksum = checkpoint.get('wordlist_checksum')
        if stored_checksum == current_checksum:
            resume_state = checkpoint
            if not args.quiet:
                existing_sentences = len(resume_state.get('sentences', []))
                remaining_words = len(resume_state.get('unused_words', []))
                print(f"\nResuming from checkpoint: {existing_sentences} sentences saved, {remaining_words} words remaining")
        else:
            if not args.quiet:
                print("\nExisting checkpoint does not match the current word list. Starting fresh.")
    elif not args.resume and checkpoint and not args.quiet:
        print("\nIgnoring checkpoint (fresh run requested with --no-resume).")
    
    if not args.resume:
        generator.clear_checkpoint()
    
    generator.initialize_word_tracking(words, resume_state=resume_state)
    stage_times['init_generator'] = time.time() - stage_start
    
    # Generate sentences (with LLM validation if enabled)
    stage_start = time.time()
    sentences = generator.generate_sentences(verbose=not args.quiet)
    if not generator.last_run_aborted:
        generator.clear_checkpoint()
    stage_times['generate_sentences'] = time.time() - stage_start
    
    # Get statistics
    stats = generator.get_statistics()
    
    return sentences, stats, generator, classifier


def main():
    """Main entry point for the sentence generator."""
    parser = argparse.ArgumentParser(
        description="Generate grammatically correct Spanish sentences from a word list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentencemaker.py --llm-filter
  python sentencemaker.py --max-words 15 --llm-filter
  python sentencemaker.py -w custom_words.txt -o custom_output.txt --llm-filter
  python sentencemaker.py --llm-filter --llm-model mistral:7b-instruct-v0.2-q4_0
        """
    )
    
    parser.add_argument(
        '-w', '--wordlist',
        default='words/words.txt',
        help='Path to file containing Spanish words (default: words/words.txt)'
    )
    parser.add_argument(
        '-o', '--output',
        default='output/sentences.txt',
        help='Output file for generated sentences (default: output/sentences.txt)'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=12,
        help='Maximum words per sentence (default: 12)'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=6,
        help='Minimum words per sentence after fixes (default: 6)'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=0,
        help='Maximum sentences to generate (default: 0, meaning all words)'
    )
    parser.add_argument(
        '--spellcheck',
        action='store_true',
        help='Enable dictionary-based spell checking (default: disabled)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed profiling to show Python vs compiled library time breakdown'
    )
    # LLM is now always required for generation - no flag needed
    parser.add_argument(
        '--llm-provider',
        choices=['ollama', 'openai', 'anthropic', 'none'],
        default='none',
        help='LLM provider for validation (default: none = no LLM validation); generation uses --gen-llm-provider'
    )
    parser.add_argument(
        '--llm-model',
        default='gemma2:9b',
        help='Model used for validation (defaults to gemma2:9b); generation uses --gen-llm-model'
    )
    parser.add_argument(
        '--validator-llm-model',
        default=None,
        help='LLM model used for validation (defaults to --llm-model when not set)'
    )
    parser.add_argument(
        '--gen-llm-provider',
        choices=['ollama', 'openai', 'anthropic'],
        default=None,
        help='LLM provider for generation (defaults to validation provider when not set)'
    )
    parser.add_argument(
        '--gen-llm-model',
        default=None,
        help='Model used for generation (defaults to validation model when not set)'
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        '--resume',
        dest='resume',
        action='store_true',
        help='Resume from checkpoint if available (default)'
    )
    resume_group.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Ignore checkpoints and start from the beginning'
    )
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("=" * 60)
        print("SentenceMaker - Spanish Sentence Generator")
        print("=" * 60)
        print("\nConfiguration:")
        print(f"  Word list: {args.wordlist}")
        print(f"  Output: {args.output}")
        print(f"  Max words/sentence: {args.max_words}")
        if args.max_sentences and args.max_sentences > 0:
            print(f"  Max sentences: {args.max_sentences}")
        else:
            print(f"  Max sentences: until all words covered")
        print(f"  NLP Model: es_core_news_sm (spaCy)")
        provider_label = {
            'ollama': 'LOCAL (FREE)',
            'openai': 'OpenAI API (COSTS $)',
            'anthropic': 'Anthropic API (COSTS $)',
            'none': 'No validation (heuristics only)'
        }
        validator_model = args.validator_llm_model or args.llm_model
        gen_provider = args.gen_llm_provider or args.llm_provider
        gen_model = args.gen_llm_model or validator_model
        print(f"  LLM Generation: {provider_label.get(gen_provider, gen_provider)} - {gen_model}")
        print(f"  LLM Validation: {provider_label.get(args.llm_provider, args.llm_provider)} - {validator_model}")
        print(f"  Spell check: {'enabled' if args.spellcheck else 'disabled'}")
        print(f"  Profiling: {'on' if args.profile else 'off'}")
        print(f"  Quiet mode: {'on' if args.quiet else 'off'}")
        print("=" * 60)
    
    # Start timing
    start_time = time.time()
    stage_times = {}
    
    # Run with or without profiling
    profiling_data = None
    run_id = load_next_run_id()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        sentences, stats, generator, classifier = run_generator(args, stage_times, run_id)
        profiler.disable()
        profiling_data = analyze_profiling_data(profiler)
    else:
        sentences, stats, generator, classifier = run_generator(args, stage_times, run_id)
    
    # Save sentences
    stage_start = time.time()
    save_sentences(sentences, args.output)
    stage_times['save_output'] = time.time() - stage_start
    
    # Final cleanup of checkpoint (normal completion)
    if not generator.last_run_aborted:
        generator.clear_checkpoint()
    
    # Save unused words if any
    if generator.unused_words:
        # Determine unused words filename based on output filename
        output_base = os.path.splitext(args.output)[0]
        unused_file = f"{output_base}_unused.txt"
        save_unused_words(generator.unused_words, classifier, unused_file)
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total words in list:     {stats['total_words']}")
        print(f"Words used:              {stats['used_words']}")
        print(f"Words not used:          {stats['unused_words']}")
        if stats.get('llm_generated', 0) > 0:
            print(f"LLM generated:           {stats['llm_generated']} sentences")
        print(f"Coverage:                {stats['coverage_percent']:.2f}%")
        print(f"Total sentences:         {len(sentences)}")
        print(f"Avg words per sentence:  {sum(len(s.split()) for s in sentences) / len(sentences):.2f}")
        print(f"Avg word usage:          {stats['avg_word_usage']:.2f}")
        print(f"Max word repetitions:    {stats['max_word_usage']}")
        
        # Show mood variety if available
        if 'mood_counts' in stats:
            print("\nSentence Variety:")
            mood_counts = stats['mood_counts']
            total = sum(mood_counts.values())
            mood_labels = {
                'Ind': 'Statements (Indicative)',
                'Imp': 'Commands (Imperative)',
                'Sub': 'Subjunctive',
                'Question': 'Questions'
            }
            for mood, count in sorted(mood_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    label = mood_labels.get(mood, mood)
                    percent = (count / total * 100) if total > 0 else 0
                    print(f"  {label:25} {count:3} ({percent:5.1f}%)")
        rejections = stats.get('rejections', {})
        if rejections:
            total_rejections = sum(rejections.values())
            heuristic = rejections.get('heuristic', 0)
            validator = rejections.get('validator', 0)
            print(f"\nRejections: {total_rejections} (heuristic={heuristic}, validator={validator})")
        template_failures = stats.get('template_failures', {})
        if template_failures:
            print("\nTop template failures:")
            for template_id, count in sorted(template_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {template_id}: {count}")
        template_successes = stats.get('template_successes', {})
        if template_successes:
            print("\nTop template selections:")
            for template_id, count in sorted(template_successes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {template_id}: {count}")
        if args.spellcheck:
            spell_unknowns = stats.get('spellcheck_unknowns', [])
            if spell_unknowns:
                print("\nTop spell-check rejects:")
                for word, count in spell_unknowns[:8]:
                    print(f"  {word:<25}{count:>5}")
            spell_total = stats.get('spellcheck_duration', 0.0)
            spell_avg = (spell_total / len(sentences)) if sentences else 0.0
            print(f"\nSpell check time:       {format_duration(spell_total)} (avg {format_duration(spell_avg)})")
        else:
            print("\nSpell check: disabled")
        print(f"\nLast sentence time: {format_duration(stats.get('last_sentence_duration', 0.0))}")
        print(f"Average sentence time: {format_duration(stats.get('avg_sentence_duration', 0.0))}")
        histogram = stats.get('duration_histogram', {})
        if histogram:
            print("Sentence duration histogram:")
            for bucket in DURATION_BUCKETS:
                if bucket in histogram:
                    print(f"  {bucket}: {histogram[bucket]}")
        print("\n" + "-" * 60)
        print("TIMING BREAKDOWN")
        print("-" * 60)
        print(f"Load word list:          {format_duration(stage_times['load_wordlist'])}")
        print(f"Initialize model:        {format_duration(stage_times['init_classifier'])}")
        print(f"Classify words:          {format_duration(stage_times['classify_words'])}")
        print(f"Initialize generator:    {format_duration(stage_times['init_generator'])}")
        print(f"Generate sentences:      {format_duration(stage_times['generate_sentences'])}")
        print(f"Save output:             {format_duration(stage_times['save_output'])}")
        print("-" * 60)
        print(f"TOTAL TIME:              {format_duration(elapsed_time)}")
        
        # Add profiling breakdown if enabled
        if profiling_data:
            print("\n" + "-" * 60)
            print("PERFORMANCE BREAKDOWN (Python vs Compiled)")
            print("-" * 60)
            print(f"Python code time:        {format_duration(profiling_data['python_time'])} ({profiling_data['python_percent']:.1f}%)")
            print(f"Compiled libraries:      {format_duration(profiling_data['compiled_time'])} ({profiling_data['compiled_percent']:.1f}%)")
            print(f"  ├─ spaCy (NLP):        {format_duration(profiling_data['spacy_time'])}")
            print(f"  └─ mlconjug3 (verbs):  {format_duration(profiling_data['mlconjug_time'])}")
            print("-" * 60)
            print(f"Profiled time total:     {format_duration(profiling_data['total_time'])}")
            print("\nNote: Compiled libraries (C/Cython) are 10-100x faster than Python")
        
        print("=" * 60)
    else:
        print(f"Generated {len(sentences)} sentences in {format_duration(elapsed_time)}")
        print(f"Coverage: {stats['coverage_percent']:.2f}% ({stats['used_words']}/{stats['total_words']} words)")
        rejections = stats.get('rejections', {})
        if rejections:
            total_rejections = sum(rejections.values())
            heuristic = rejections.get('heuristic', 0)
            validator = rejections.get('validator', 0)
            print(f"Rejections: {total_rejections} (heuristic={heuristic}, validator={validator})")
        template_failures = stats.get('template_failures', {})
        if template_failures:
            print("Top template failures:")
            for template_id, count in sorted(template_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {template_id}: {count}")
        template_successes = stats.get('template_successes', {})
        if template_successes:
            print("Top template selections:")
            for template_id, count in sorted(template_successes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {template_id}: {count}")
        if args.spellcheck:
            spell_unknowns = stats.get('spellcheck_unknowns', [])
            if spell_unknowns:
                print("Top spell-check rejects:")
                for word, count in spell_unknowns[:8]:
                    print(f"  {word:<25}{count:>5}")
            spell_total = stats.get('spellcheck_duration', 0.0)
            spell_avg = (spell_total / len(sentences)) if sentences else 0.0
            print(f"Spell check time: {format_duration(spell_total)} (avg {format_duration(spell_avg)})")
        else:
            print("Spell check: disabled")
        print(f"Last sentence time: {format_duration(stats.get('last_sentence_duration', 0.0))}")
        print(f"Average sentence time: {format_duration(stats.get('avg_sentence_duration', 0.0))}")
        histogram = stats.get('duration_histogram', {})
        if histogram:
            print("Sentence duration histogram:")
            for bucket in DURATION_BUCKETS:
                if bucket in histogram:
                    print(f"  {bucket}: {histogram[bucket]}")
            legacy_count = histogram.get(LEGACY_DURATION_BUCKET)
            if legacy_count:
                print(f"  {LEGACY_DURATION_BUCKET} (legacy): {legacy_count}")


if __name__ == "__main__":
    main()
