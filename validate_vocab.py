#!/usr/bin/env python3
"""
Vocabulary File Validation Script
Checks for common issues in the vocabulary file.
"""

import re
from collections import Counter

def validate_vocabulary_file(filename):
    """Validate vocabulary file for common issues."""
    
    issues = []
    warnings = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Validating {len(lines)} words in {filename}...")
    print()
    
    # Check 1: Concatenated words (lowercase followed by uppercase)
    print("🔍 Checking for concatenated words...")
    concatenated = []
    for i, word in enumerate(lines, 1):
        # Pattern: lowercase letter followed by uppercase (e.g., "wordWord")
        if re.search(r'[a-záéíóúñü][A-ZÁÉÍÓÚÑÜ]', word):
            concatenated.append((i, word))
    
    if concatenated:
        issues.append(f"Found {len(concatenated)} concatenated words:")
        for line_num, word in concatenated[:10]:  # Show first 10
            issues.append(f"  Line {line_num}: {word}")
        if len(concatenated) > 10:
            issues.append(f"  ... and {len(concatenated) - 10} more")
    else:
        print("  ✓ No concatenated words found")
    
    # Check 2: Duplicates
    print("🔍 Checking for duplicates...")
    word_counts = Counter(lines)
    duplicates = [(word, count) for word, count in word_counts.items() if count > 1]
    
    if duplicates:
        warnings.append(f"Found {len(duplicates)} duplicate words:")
        for word, count in sorted(duplicates, key=lambda x: -x[1])[:10]:
            warnings.append(f"  '{word}' appears {count} times")
        if len(duplicates) > 10:
            warnings.append(f"  ... and {len(duplicates) - 10} more")
    else:
        print("  ✓ No duplicates found")
    
    # Check 3: Very long words (potential concatenations)
    print("🔍 Checking for suspiciously long words...")
    long_words = [(i, word) for i, word in enumerate(lines, 1) if len(word) > 25]
    
    if long_words:
        warnings.append(f"Found {len(long_words)} very long words (>25 chars):")
        for line_num, word in long_words[:10]:
            warnings.append(f"  Line {line_num}: {word} ({len(word)} chars)")
        if len(long_words) > 10:
            warnings.append(f"  ... and {len(long_words) - 10} more")
    else:
        print("  ✓ No suspiciously long words found")
    
    # Check 4: Empty lines or whitespace issues
    print("🔍 Checking for whitespace issues...")
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    
    whitespace_issues = []
    for i, line in enumerate(raw_lines, 1):
        if line.strip() != line.rstrip('\n'):
            whitespace_issues.append(i)
    
    if whitespace_issues:
        warnings.append(f"Found {len(whitespace_issues)} lines with leading/trailing whitespace")
    else:
        print("  ✓ No whitespace issues found")
    
    # Check 5: Non-alphabetic characters (except allowed ones)
    print("🔍 Checking for unusual characters...")
    allowed_pattern = re.compile(r'^[a-záéíóúñüA-ZÁÉÍÓÚÑÜ¿?¡!.,\-/\s]+$')
    unusual = []
    for i, word in enumerate(lines, 1):
        if not allowed_pattern.match(word):
            unusual.append((i, word))
    
    if unusual:
        warnings.append(f"Found {len(unusual)} words with unusual characters:")
        for line_num, word in unusual[:10]:
            warnings.append(f"  Line {line_num}: {word}")
        if len(unusual) > 10:
            warnings.append(f"  ... and {len(unusual) - 10} more")
    else:
        print("  ✓ No unusual characters found")
    
    # Print results
    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print()
    
    if not issues and not warnings:
        print("✅ All checks passed! Vocabulary file looks good.")
        return True
    
    if issues:
        print("❌ CRITICAL ISSUES FOUND:")
        print()
        for issue in issues:
            print(issue)
        print()
    
    if warnings:
        print("⚠️  WARNINGS:")
        print()
        for warning in warnings:
            print(warning)
        print()
    
    if issues:
        print("❌ Validation FAILED - please fix critical issues")
        return False
    else:
        print("⚠️  Validation passed with warnings")
        return True

if __name__ == '__main__':
    import sys
    
    filename = 'words/words.txt'
    success = validate_vocabulary_file(filename)
    
    sys.exit(0 if success else 1)
