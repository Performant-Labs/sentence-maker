#!/usr/bin/env python3
"""
Vocabulary File Cleanup Script
Fixes concatenated words and removes duplicates while preserving order.
"""

import re
from collections import OrderedDict

# Known concatenated words to fix (detected patterns)
CONCATENATED_FIXES = {
    'InfoisinfoEmpresas': ['Info', 'es', 'info', 'Empresas'],
    'JurídicaJuvenil': ['Jurídica', 'Juvenil'],
    'PartesPatronato': ['Partes', 'Patronato'],
    'AdBlue': ['Ad', 'Blue'],  # Could be a brand name, but likely concatenated
    'FederalHace': ['Federal', 'Hace'],
    'SeránSería': ['Serán', 'Sería'],
    'CuartaCuentan': ['Cuarta', 'Cuentan'],
    'InfoisinfoBoulevard': ['Info', 'es', 'info', 'Boulevard'],
    'PúblicaGobierno': ['Pública', 'Gobierno'],
    'espírituestablecido': ['espíritu', 'establecido'],
}

def fix_vocabulary_file(input_file, output_file):
    """Fix concatenated words and remove duplicates."""
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Original word count: {len(lines)}")
    
    # Fix concatenated words
    fixed_lines = []
    fixes_applied = 0
    
    for line in lines:
        if line in CONCATENATED_FIXES:
            # Replace with split words
            fixed_words = CONCATENATED_FIXES[line]
            fixed_lines.extend(fixed_words)
            fixes_applied += 1
            print(f"  Fixed: {line} → {', '.join(fixed_words)}")
        else:
            fixed_lines.append(line)
    
    print(f"\nConcatenated words fixed: {fixes_applied}")
    print(f"Word count after fixes: {len(fixed_lines)}")
    
    # Remove duplicates while preserving order (case-sensitive)
    seen = set()
    unique_lines = []
    duplicates_removed = 0
    
    for line in fixed_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
        else:
            duplicates_removed += 1
            print(f"  Removed duplicate: {line}")
    
    print(f"\nDuplicates removed: {duplicates_removed}")
    print(f"Final word count: {len(unique_lines)}")
    
    # Write cleaned file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    print(f"\n✓ Cleaned vocabulary saved to: {output_file}")
    
    return len(lines), len(unique_lines), fixes_applied, duplicates_removed

if __name__ == '__main__':
    input_file = 'words/words.txt'
    output_file = 'words/words.txt'
    
    print("=" * 60)
    print("VOCABULARY FILE CLEANUP")
    print("=" * 60)
    print()
    
    original, final, fixes, dupes = fix_vocabulary_file(input_file, output_file)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original words:       {original}")
    print(f"Concatenations fixed: {fixes}")
    print(f"Duplicates removed:   {dupes}")
    print(f"Final words:          {final}")
    print(f"Net change:           {final - original:+d}")
    print()
