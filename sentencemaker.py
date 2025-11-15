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


def run_generator(args, stage_times):
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
        print(f"Loaded {len(words)} words ({stage_times['load_wordlist']:.3f}s)")
    
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
        print(f"Model loaded ({stage_times['init_classifier']:.3f}s)")
    
    # Classify words
    stage_start = time.time()
    classifier.classify_words(words)
    stage_times['classify_words'] = time.time() - stage_start
    
    if not args.quiet:
        print(f"Classification time: {stage_times['classify_words']:.3f}s")
    
    # Initialize LLM (required for generation)
    llm_validator = None
    if True:  # LLM is always required
        from llm_validator import LLMValidator
        
        # Show provider info
        if not args.quiet:
            provider_info = {
                'ollama': f'LOCAL (FREE) - {args.llm_model}',
                'openai': f'OpenAI API (COSTS $) - {args.llm_model}',
                'anthropic': f'Anthropic API (COSTS $) - {args.llm_model}'
            }
            print(f"\nInitializing LLM validator: {provider_info[args.llm_provider]}")
            if args.llm_provider != 'ollama':
                print(f"  ⚠️  WARNING: Using API provider - this will incur costs!")
                print(f"  Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        
        stage_start = time.time()
        try:
            # Use same model for validation to ensure quality
            validator_model = args.llm_model
            if not args.quiet and args.llm_model in ["gemma2:27b", "gemma2:9b"]:
                print(f"  Using {validator_model} for both generation and validation (best quality)")
            
            llm_validator = LLMValidator(model=validator_model, provider=args.llm_provider)
            stage_times['init_llm'] = time.time() - stage_start
            
            if not args.quiet:
                print(f"LLM validator ready ({stage_times['init_llm']:.3f}s)")
        except (ConnectionError, TimeoutError, ValueError) as e:
            print(f"\n❌ Error: {e}")
            print("\nCannot continue without LLM.")
            print("Please start ollama: ollama serve")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error initializing LLM validator: {e}")
            sys.exit(1)
    
    # Initialize generator with output file for incremental saving
    stage_start = time.time()
    if not args.quiet:
        print(f"\nOutput will be saved to: {args.output}")
        print("(File will be updated after each sentence)")
        print("(Using LLM for generation and validation)")
    generator = SentenceGenerator(classifier, max_words=args.max_words, output_file=args.output, llm_validator=llm_validator, generation_model=args.llm_model, max_sentences=args.max_sentences)
    
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
        default=15,
        help='Maximum words per sentence (default: 15)'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=0,
        help='Maximum sentences to generate (default: 0, meaning all words)'
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
        choices=['ollama', 'openai', 'anthropic'],
        default='ollama',
        help='LLM provider: ollama (local, FREE), openai (API, costs $), anthropic (API, costs $) (default: ollama)'
    )
    parser.add_argument(
        '--llm-model',
        default='gemma2:9b',
        help='Model name - ollama: gemma2:9b (default, best balance), gemma2:27b (highest quality), mistral:7b-instruct-v0.2-q4_0 (faster) | openai: gpt-4o-mini, gpt-4 | anthropic: claude-3-haiku, claude-3-sonnet'
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
        print(f"  NLP Model: es_core_news_sm (spaCy)")
        provider_label = {
            'ollama': 'LOCAL (FREE)',
            'openai': 'OpenAI API (COSTS $)',
            'anthropic': 'Anthropic API (COSTS $)'
        }
        print(f"  LLM Validation: {provider_label[args.llm_provider]} - {args.llm_model}")
        print("=" * 60)
    
    # Start timing
    start_time = time.time()
    stage_times = {}
    
    # Run with or without profiling
    profiling_data = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        sentences, stats, generator, classifier = run_generator(args, stage_times)
        profiler.disable()
        profiling_data = analyze_profiling_data(profiler)
    else:
        sentences, stats, generator, classifier = run_generator(args, stage_times)
    
    # Save sentences
    stage_start = time.time()
    save_sentences(sentences, args.output)
    stage_times['save_output'] = time.time() - stage_start
    
    # Final cleanup of checkpoint (normal completion)
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
        print(f"\nLast sentence time: {stats.get('last_sentence_duration', 0.0):.2f}s")
        print(f"Average sentence time: {stats.get('avg_sentence_duration', 0.0):.2f}s")
        histogram = stats.get('duration_histogram', {})
        if histogram:
            print("Sentence duration histogram:")
            for bucket in ["<1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-6s", ">=6s"]:
                if bucket in histogram:
                    print(f"  {bucket}: {histogram[bucket]}")
        print("\n" + "-" * 60)
        print("TIMING BREAKDOWN")
        print("-" * 60)
        print(f"Load word list:          {stage_times['load_wordlist']:.3f}s")
        print(f"Initialize model:        {stage_times['init_classifier']:.3f}s")
        print(f"Classify words:          {stage_times['classify_words']:.3f}s")
        print(f"Initialize generator:    {stage_times['init_generator']:.3f}s")
        print(f"Generate sentences:      {stage_times['generate_sentences']:.3f}s")
        print(f"Save output:             {stage_times['save_output']:.3f}s")
        print("-" * 60)
        print(f"TOTAL TIME:              {elapsed_time:.3f}s")
        
        # Add profiling breakdown if enabled
        if profiling_data:
            print("\n" + "-" * 60)
            print("PERFORMANCE BREAKDOWN (Python vs Compiled)")
            print("-" * 60)
            print(f"Python code time:        {profiling_data['python_time']:.3f}s ({profiling_data['python_percent']:.1f}%)")
            print(f"Compiled libraries:      {profiling_data['compiled_time']:.3f}s ({profiling_data['compiled_percent']:.1f}%)")
            print(f"  ├─ spaCy (NLP):        {profiling_data['spacy_time']:.3f}s")
            print(f"  └─ mlconjug3 (verbs):  {profiling_data['mlconjug_time']:.3f}s")
            print("-" * 60)
            print(f"Profiled time total:     {profiling_data['total_time']:.3f}s")
            print("\nNote: Compiled libraries (C/Cython) are 10-100x faster than Python")
        
        print("=" * 60)
    else:
        print(f"Generated {len(sentences)} sentences in {elapsed_time:.2f}s")
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
        print(f"Last sentence time: {stats.get('last_sentence_duration', 0.0):.2f}s")
        print(f"Average sentence time: {stats.get('avg_sentence_duration', 0.0):.2f}s")
        histogram = stats.get('duration_histogram', {})
        if histogram:
            print("Sentence duration histogram:")
            for bucket in ["<1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-6s", ">=6s"]:
                if bucket in histogram:
                    print(f"  {bucket}: {histogram[bucket]}")


if __name__ == "__main__":
    main()
