"""
Sentence Generator Module
Generates grammatically correct Spanish sentences using LLM with template guidance.
"""
import json
import hashlib
import os
import random
import re
import time
import sys
from typing import List, Optional
from collections import defaultdict
from word_classifier import WordClassifier
from sentence_templates import TemplateLibrary, Template


class SentenceGenerator:
    """Generates Spanish sentences to cover all words in a vocabulary."""
    
    COLOR_WORDS = {
        'rojo', 'rojos', 'roja', 'rojas',
        'blanco', 'blancos', 'blanca', 'blancas',
        'negro', 'negros', 'negra', 'negras',
        'verde', 'verdes', 'amarillo', 'amarilla', 'amarillos', 'amarillas',
        'azul', 'azules', 'morado', 'morado', 'morada', 'moradas',
        'gris', 'grises'
    }
    COLOR_CONTEXT_WORDS = {
        'casa', 'hogar', 'pared', 'paredes', 'habitacion', 'habitación', 'puerta',
        'ventana', 'techo', 'sala', 'sillón', 'sillon', 'auto', 'automóvil', 'automovil',
        'vehículo', 'vehiculo', 'coche', 'camioneta', 'camión', 'camion', 'camisa', 'camiseta',
        'vestido', 'falda', 'zapato', 'zapatos', 'ropa', 'sombrero', 'mesa', 'silla', 'sofá',
        'sofa', 'carro', 'bicicleta', 'flor', 'flores', 'árbol', 'arbol', 'fruta', 'frutas',
        'manzana', 'sandía', 'sandia', 'cama', 'colcha', 'cocina', 'plato', 'comida',
        'bebida', 'taza', 'cuadro', 'decoración', 'decoracion'
    }
    ABSTRACT_COLOR_WORDS = {
        'sentimiento', 'sentimientos', 'emocion', 'emoción', 'emociones', 'temporada',
        'sensación', 'sensacion', 'psicologia', 'psicología', 'nombramiento', 'nombramientos',
        'signos', 'reglamento', 'reglamentos', 'salud', 'tiempo', 'rutina'
    }
    FANTASY_WORDS = {
        'fantasma', 'fantasmas', 'vampiro', 'vampiros', 'zombi', 'zombie', 'zombis',
        'zombies', 'monstruo', 'monstruos', 'dragón', 'dragon', 'dragones',
        'bruja', 'brujas', 'duende', 'duendes', 'ratón gigante', 'raton gigante'
    }
    
    # Seed topics with associated keywords for smart topic selection
    SEED_TOPICS = {
        "daily routine": ["mañana", "tarde", "noche", "día", "hora", "tiempo", "siempre", "nunca", "ahora"],
        "family life": ["familia", "madre", "padre", "hijo", "hija", "hermano", "hermana", "abuelo", "abuela", "niño", "bebé"],
        "work": ["trabajo", "oficina", "jefe", "empleado", "sindicato", "empresa", "negocio", "profesional", "carrera"],
        "home": ["casa", "hogar", "cocina", "habitación", "sala", "jardín", "puerta", "ventana", "techo"],
        "food": ["come", "comer", "comida", "cocina", "cocinar", "restaurante", "plato", "delicioso", "hambre"],
        "nature": ["río", "montaña", "árbol", "bosque", "animal", "gato", "perro", "pájaro", "flor", "planta"],
        "education": ["estudia", "estudiar", "escuela", "universidad", "libro", "manuscrito", "profesor", "alumno", "clase"],
        "emotions": ["feliz", "triste", "amor", "libertad", "alegre", "contento", "emoción", "sentimiento", "corazón"],
        "travel": ["viaje", "viajar", "ciudad", "país", "lugar", "mundo", "camino", "calle", "lejos", "cerca"],
        "health": ["salud", "médico", "hospital", "enfermo", "sano", "cuerpo", "dolor", "medicina"],
        "leisure": ["juega", "jugar", "diversión", "fiesta", "música", "baile", "amigo", "paseo"],
        "shopping": ["compra", "comprar", "tienda", "mercado", "precio", "dinero", "pagar"]
    }
    
    def __init__(self, classifier: WordClassifier, max_words: int = 10, output_file: str = None, llm_validator=None, generation_model: str = None):
        """
        Initialize the sentence generator.
        
        Args:
            classifier: WordClassifier instance with categorized words
            max_words: Maximum words per sentence
            output_file: Optional path to save sentences incrementally
            llm_validator: LLMValidator for semantic coherence checking (required)
            generation_model: Model to use for generation (if different from validator)
        """
        self.classifier = classifier
        self.max_words = max_words
        self.output_file = output_file
        self.llm_validator = llm_validator
        self.generation_model = generation_model or (llm_validator.model if llm_validator else None)
        self.template_library = TemplateLibrary()
        self.used_words = set()
        self.unused_words = set()
        self.word_usage_count = defaultdict(int)
        self.llm_generated_count = 0
        self.recent_sentence_starts = []  # Track recent sentence beginnings to avoid repetition
        self.sentences = []
        self.wordlist_checksum = None
        self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._recent_template_patterns = []
        self.imperative_forms = set()
        self.rejection_counts = defaultdict(int)
        self.stats_file = None
        self.template_fail_counts = defaultdict(int)
        self.last_sentence_duration = 0.0
        self.total_sentence_duration = 0.0
        self.successful_sentences = 0
        
    def initialize_word_tracking(self, all_words: List[str], resume_state: Optional[dict] = None):
        """
        Initialize tracking for which words have been used.
        
        Args:
            all_words: Complete list of words to track
            resume_state: Optional checkpoint data to resume progress
        """
        self.wordlist_checksum = self.compute_wordlist_checksum(all_words)
        self._build_imperative_forms()
        if self.output_file:
            self.stats_file = f"{self.output_file}.stats.txt"
        else:
            self.stats_file = "sentencemaker_stats.txt"
        
        if resume_state:
            self._restore_from_checkpoint(resume_state, all_words)
        else:
            self._start_fresh(all_words)
        self._write_stats_snapshot()
    
    def _start_fresh(self, all_words: List[str]):
        """Reset all tracking structures for a new run."""
        self.unused_words = set(all_words)
        self.used_words = set()
        self.word_usage_count = defaultdict(int)
        self.sentences = []
        self.llm_generated_count = 0
        self.recent_sentence_starts = []
        self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._recent_template_patterns = []
        self.rejection_counts = defaultdict(int)
        self.rejection_counts = defaultdict(int)
        
        if self.output_file:
            try:
                output_dir = os.path.dirname(self.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(self.output_file, 'w', encoding='utf-8'):
                    pass
            except Exception:
                pass  # Don't crash if we can't clear the file
            self.clear_checkpoint()
    
    def _restore_from_checkpoint(self, state: dict, all_words: List[str]):
        """Restore tracking state from a checkpoint."""
        all_words_set = set(all_words)
        stored_used = set(state.get('used_words', []))
        stored_unused = set(state.get('unused_words', []))
        
        # Drop words not in the current list and add any missing ones back to unused
        used_words = stored_used & all_words_set
        unused_words = stored_unused & all_words_set
        missing_words = all_words_set - used_words - unused_words
        unused_words |= missing_words
        
        self.used_words = used_words
        self.unused_words = unused_words
        usage_counts = {
            word: count
            for word, count in state.get('word_usage_count', {}).items()
            if word in all_words_set
        }
        self.word_usage_count = defaultdict(int, usage_counts)
        self.sentences = list(state.get('sentences', []))
        self.llm_generated_count = state.get('llm_generated_count', 0)
        self.recent_sentence_starts = state.get('recent_sentence_starts', [])
        mood_counts = state.get('mood_counts')
        if mood_counts:
            self._mood_counts = mood_counts
        else:
            self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._recent_template_patterns = state.get('template_patterns', [])
        self.rejection_counts = defaultdict(int, state.get('rejection_counts', {}))
        self.template_fail_counts = defaultdict(int, state.get('template_failures', {}))
        self.last_sentence_duration = state.get('last_sentence_duration', 0.0)
        self.total_sentence_duration = state.get('total_sentence_duration', 0.0)
        self.successful_sentences = state.get('successful_sentences', len(self.sentences))
        
        if self.output_file:
            # Ensure checkpoint output exists and reflects current sentences
            self._save_sentences_incremental(self.sentences)
    
    @staticmethod
    def compute_wordlist_checksum(words: List[str]) -> str:
        """Compute a checksum for the provided word list (order-independent)."""
        normalized = '\n'.join(sorted(words))
        return hashlib.sha1(normalized.encode('utf-8')).hexdigest()

    def _build_imperative_forms(self):
        """Collect known imperative forms from classifier data."""
        self.imperative_forms = set()
        for word, info in self.classifier.word_info.items():
            if info.get('mood') == 'Imp':
                self.imperative_forms.add(word.lower())
        self.imperative_forms.update({
            'pon', 'ponte', 'ponga', 'pongamos', 'comparte', 'comparta',
            'ordena', 'ordene', 'organiza', 'organice', 'anota', 'anote',
            'cuenta', 'cuente', 'busca', 'busque', 'mira', 'mire', 'entra',
            'entre', 'ingresa', 'ingrese', 'ofrece', 'ofrezca', 'cuida', 'cuide',
            'sigue', 'siga', 'apoya', 'apoye', 'revisa', 'revise', 'permite',
            'permita', 'recibe', 'reciba', 'desea', 'desee', 'evita', 'evite',
            'respeta', 'respete', 'ajusta', 'ajuste', 'ordenen', 'solicita',
            'solicite'
        })
    
    def _save_sentences_incremental(self, sentences: List[str]):
        """
        Save sentences incrementally to output file.
        
        Args:
            sentences: List of all sentences generated so far
        """
        try:
            import os
            # Create output directory if needed
            output_dir = os.path.dirname(self.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Write all sentences (overwrite file each time)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
        except Exception as e:
            # Don't crash if save fails, just continue
            print(f"\n  Warning: Could not save incrementally: {e}")
            pass
    
    def _get_checkpoint_path(self) -> Optional[str]:
        """Return the path to the checkpoint file if output is specified."""
        if not self.output_file:
            return None
        return f"{self.output_file}.state.json"
    
    def _save_checkpoint(self):
        """Persist generator state so runs can resume later."""
        path = self._get_checkpoint_path()
        if not path:
            return
        
        state = {
            'sentences': self.sentences,
            'used_words': sorted(self.used_words),
            'unused_words': sorted(self.unused_words),
            'word_usage_count': dict(self.word_usage_count),
            'llm_generated_count': self.llm_generated_count,
            'recent_sentence_starts': self.recent_sentence_starts[-40:],
            'mood_counts': self._mood_counts,
            'wordlist_checksum': self.wordlist_checksum,
            'max_words': self.max_words,
            'template_patterns': self._recent_template_patterns[-40:],
            'rejection_counts': dict(self.rejection_counts),
            'template_failures': dict(self.template_fail_counts),
            'last_sentence_duration': self.last_sentence_duration,
            'total_sentence_duration': self.total_sentence_duration,
            'successful_sentences': self.successful_sentences,
            'timestamp': time.time()
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Ignore checkpoint errors

    def _get_stats_snapshot(self) -> str:
        """Return a human-readable snapshot of current stats."""
        stats = self.get_statistics()
        lines = [
            "SentenceMaker Stats Snapshot",
            "=============================",
            f"Sentences generated : {len(self.sentences)}",
            f"Words used          : {stats['used_words']}",
            f"Words remaining     : {stats['unused_words']}",
            f"Coverage            : {stats['coverage_percent']:.2f}%",
            f"Last sentence time  : {stats['last_sentence_duration']:.2f}s",
            f"Avg sentence time   : {stats['avg_sentence_duration']:.2f}s",
            ""
        ]
        rejections = stats.get('rejections', {})
        if rejections:
            lines.append(f"Total rejections    : {rejections.get('heuristic', 0) + rejections.get('validator', 0)}")
            lines.append(f"  - Heuristic       : {rejections.get('heuristic', 0)}")
            lines.append(f"  - Validator       : {rejections.get('validator', 0)}")
            detail_keys = sorted(k for k in rejections if k.startswith('heuristic_'))
            if detail_keys:
                lines.append("")
                lines.append("Heuristic breakdown:")
                for key in detail_keys:
                    label = key.replace('heuristic_', '').replace('_', ' ')
                    lines.append(f"  - {label}: {rejections[key]}")
        else:
            lines.append("Total rejections    : 0")
        if self.template_fail_counts:
            lines.append("")
            lines.append("Top template failures:")
            sorted_templates = sorted(self.template_fail_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for template_id, count in sorted_templates:
                lines.append(f"  - {template_id}: {count}")
        lines.append("")
        lines.append("Updated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        return "\n".join(lines)

    def _write_stats_snapshot(self):
        """Persist current stats snapshot for inspection."""
        if not self.stats_file:
            return
        try:
            snapshot = self._get_stats_snapshot()
            output_dir = os.path.dirname(self.stats_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                f.write(snapshot + "\n")
        except Exception:
            pass
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint data if it exists."""
        path = self._get_checkpoint_path()
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"\n  Warning: Could not read checkpoint {path}: {e}")
            return None
    
    def clear_checkpoint(self):
        """Remove the checkpoint file."""
        path = self._get_checkpoint_path()
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        
    def get_unused_words_by_pos(self, pos: str) -> List[str]:
        """
        Get unused words of a specific POS.
        
        Args:
            pos: Part of speech tag
            
        Returns:
            List of unused words with that POS
        """
        unused = [w for w in self.unused_words if self.classifier.get_word_info(w).get('pos') == pos]
        return unused
    
    
    def mark_word_used(self, word: str):
        """Mark a word as used."""
        if word in self.unused_words:
            self.unused_words.remove(word)
            self.used_words.add(word)
        self.word_usage_count[word] += 1
    
    def _select_seed_topic(self, available_words: dict) -> Optional[str]:
        """
        Select a seed topic based on available words.
        
        Args:
            available_words: Dictionary of {POS: [words]} available for sentence
            
        Returns:
            Seed topic string or None for no seed
        """
        # Flatten all available words
        all_words = []
        for words in available_words.values():
            all_words.extend(words)
        
        # Score each topic based on keyword matches
        topic_scores = {}
        for topic, keywords in self.SEED_TOPICS.items():
            score = sum(1 for word in all_words if word.lower() in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # If we have matching topics, pick the best one
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            # Only use seed if we have at least 2 matching words
            if topic_scores[best_topic] >= 2:
                return best_topic
        
        # Otherwise, randomly decide whether to use a seed (50% chance)
        if random.random() < 0.5:
            return random.choice(list(self.SEED_TOPICS.keys()))
        
        return None
    
    def _generate_with_llm(self, template: Template, mark_used: bool = True) -> Optional[tuple[str, List[str]]]:
        """
        Ask LLM to generate a sentence when template-based generation fails.
        
        Args:
            template: Template to use as guide
            mark_used: Whether to mark words as used
            
        Returns:
            Tuple of (sentence, list of words used) or None if failed
        """
        if not self.llm_validator:
            return None
        
        # Get unused words by POS needed for template
        available_words = {}
        for pos in set(template.slots):
            words = self.get_unused_words_by_pos(pos)
            if not words:
                # If no unused words, try all words
                words = self.classifier.get_words_by_pos(pos)
            if words:
                available_words[pos] = words[:20]  # Limit to 20 per POS
        
        if not available_words:
            return None
        
        # Select seed topic based on available words
        seed_topic = self._select_seed_topic(available_words)
        
        # Build prompt based on template mood
        mood_instructions = {
            'Ind': 'Create a declarative statement',
            'Imp': 'Create an imperative command (use command form)',
            'Sub': 'Create a sentence with subjunctive mood (e.g., "Espero que...", "Quiero que...")'
        }
        
        mood_instruction = mood_instructions.get(template.mood, 'Create a declarative statement')
        
        # Check for negative imperative (requires subjunctive in Spanish)
        has_negative_imperative = template.mood == 'Imp' and 'no' in template.glue_words
        if has_negative_imperative:
            mood_instruction = 'Create a negative imperative command (use subjunctive mood after "no", e.g., "¡No comas!", "¡No corras!")'
        
        # Check if template has question words
        has_question = any(q in template.glue_words for q in ['¿qué', '¿dónde', '¿cuándo', '¿cómo', '¿quién', '¿por qué'])
        if has_question:
            mood_instruction = 'Create a question'
        
        # Add structural guidance based on template complexity
        structure_hints = []
        
        # Check for connectors/conjunctions
        if 'y' in template.glue_words:
            structure_hints.append('Use "y" to connect two clauses')
        if 'pero' in template.glue_words:
            structure_hints.append('Use "pero" to contrast two ideas')
        if 'o' in template.glue_words:
            structure_hints.append('Use "o" to present alternatives')
        if 'porque' in template.glue_words:
            structure_hints.append('Use "porque" to explain a reason')
        if 'cuando' in template.glue_words:
            structure_hints.append('Use "cuando" for temporal relationship')
        if 'si' in template.glue_words:
            structure_hints.append('Use "si" for conditional')
        if 'aunque' in template.glue_words:
            structure_hints.append('Use "aunque" for concession')
        if any(q in template.glue_words for q in ['que', 'quien', 'donde']):
            if template.mood == 'Sub':
                structure_hints.append('Use "que" with subjunctive clause')
        
        # Indicate complexity level
        num_slots = len(template.slots)
        if num_slots >= 6:
            structure_hints.append('Create a complex sentence with multiple clauses')
        elif num_slots >= 4:
            structure_hints.append('Create a compound sentence')
        
        # Build base instruction with optional seed
        if seed_topic:
            base_instruction = f"{mood_instruction} about {seed_topic} in Spanish using ONLY words from this list:"
        else:
            base_instruction = f"{mood_instruction} in Spanish using ONLY words from this list:"
        
        prompt = f"""⚠️ CRITICAL: Maximum {self.max_words} words. Count carefully.

{base_instruction}

"""
        for pos, words in available_words.items():
            prompt += f"{pos}: {', '.join(words)}\n"
        
        prompt += f"""
Requirements:
- Maximum {self.max_words} words total
- Use as many words from the list as fit naturally (skip 1-2 if needed)
- Sentence should make reasonable sense (avoid completely random combinations)
- MUST be grammatically correct: proper gender agreement, correct verb conjugation, proper word order
- DO NOT start with "Espero que" - use varied sentence structures"""
        
        # Add warning about recent patterns
        if self.recent_sentence_starts:
            recent_starts = list(set(self.recent_sentence_starts[-10:]))  # Last 10 unique starts
            if len(recent_starts) > 0:
                prompt += f"\n- CRITICAL: DO NOT start with any of these recently used patterns (will be rejected): {', '.join(recent_starts)}"
        
        prompt += """
- Make it sound natural to a Spanish speaker
- DO NOT use markdown formatting (no **, __, *, or _ characters)
- DO NOT add extra spaces (use single spaces only)"""
        
        # Add structure hints if any
        if structure_hints:
            prompt += "\n- " + "\n- ".join(structure_hints)
        
        prompt += """
- Return ONLY the plain sentence text, no formatting or markup

Sentence:"""
        
        try:
            # Try up to 2 times to get a grammatically correct sentence
            # (Reduced from 3 to avoid slowdowns with large models)
            max_attempts = 2
            for attempt in range(max_attempts):
                # Ask LLM to generate
                if hasattr(self.llm_validator, '_check_ollama'):
                    import requests
                    payload = {
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5 + (attempt * 0.15),  # Start lower for better instruction following
                            "num_predict": 50  # Reduced to discourage long outputs
                        }
                    }
                    response = requests.post(self.llm_validator.url, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    sentence = result['response'].strip()
                else:
                    # For API providers, would need different implementation
                    return None
                
                # Clean up sentence - extract only the first line (ignore LLM chatter)
                sentence = sentence.strip().strip('"').strip("'")
                # Take only the first line if LLM added extra text
                if '\n' in sentence:
                    sentence = sentence.split('\n')[0].strip()
                # Remove common LLM artifacts
                if sentence.lower().startswith('sentence:'):
                    sentence = sentence[9:].strip()
                # Remove markdown formatting (bold, italic, etc.)
                sentence = re.sub(r'\*\*([^*]+)\*\*', r'\1', sentence)  # **bold**
                sentence = re.sub(r'\*([^*]+)\*', r'\1', sentence)      # *italic*
                sentence = re.sub(r'__([^_]+)__', r'\1', sentence)      # __bold__
                sentence = re.sub(r'_([^_]+)_', r'\1', sentence)        # _italic_
                # Remove extra spaces
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                
                # If LLM generated multiple sentences, take only the first one
                # Split on period, exclamation, or question mark followed by space
                sentences_split = re.split(r'[.!?]\s+', sentence)
                if len(sentences_split) > 1:
                    sentence = sentences_split[0]
                
                if not sentence:
                    continue  # Try again
                
                # Check word count BEFORE validation (reject if too long)
                word_count = len(sentence.split())
                if word_count > self.max_words:
                    # Sentence is too long, try again
                    continue
                
                # Format sentence
                formatted_sentence = self._format_sentence(sentence, template)
                
                # Check for repetitive starts BEFORE validation (save time)
                words_in_sentence = formatted_sentence.split()
                if len(words_in_sentence) >= 5:
                    sentence_start = ' '.join(words_in_sentence[:5])
                elif len(words_in_sentence) >= 4:
                    sentence_start = ' '.join(words_in_sentence[:4])
                elif len(words_in_sentence) >= 3:
                    sentence_start = ' '.join(words_in_sentence[:3])
                else:
                    sentence_start = None
                
                # Reject if this start was used recently (within last 15 sentences)
                if sentence_start and self.recent_sentence_starts:
                    recent_15 = self.recent_sentence_starts[-15:]
                    if sentence_start in recent_15:
                        continue  # Skip this sentence, try again
                
                # Validate grammar using LLM validator
                if not self._passes_semantic_heuristics(formatted_sentence, template):
                    continue
                
                if not self._passes_semantic_heuristics(formatted_sentence, template):
                    continue
                
                if self.llm_validator and self.llm_validator.is_coherent(formatted_sentence):
                    # Track sentence start to avoid repetition (first 5 words)
                    words_in_sentence = formatted_sentence.split()
                    if len(words_in_sentence) >= 5:
                        sentence_start = ' '.join(words_in_sentence[:5])
                    elif len(words_in_sentence) >= 4:
                        sentence_start = ' '.join(words_in_sentence[:4])
                    elif len(words_in_sentence) >= 3:
                        sentence_start = ' '.join(words_in_sentence[:3])
                    else:
                        sentence_start = ' '.join(words_in_sentence[:2]) if len(words_in_sentence) >= 2 else ""
                    
                    if sentence_start:
                        self.recent_sentence_starts.append(sentence_start)
                        # Keep only last 40 starts (longer memory)
                        if len(self.recent_sentence_starts) > 40:
                            self.recent_sentence_starts.pop(0)
                    
                    # Extract words used from sentence
                    words_used = []
                    sentence_lower = sentence.lower()
                    
                    for pos, words in available_words.items():
                        for word in words:
                            if word.lower() in sentence_lower:
                                words_used.append(word)
                    
                    # Mark words as used if requested
                    if mark_used:
                        for word in words_used:
                            self.mark_word_used(word)
                        
                        # Track that this was LLM-generated
                        self.llm_generated_count += 1
                        pattern = self._template_pattern_signature(template)
                        if pattern:
                            self._recent_template_patterns.append(pattern)
                            if len(self._recent_template_patterns) > 40:
                                self._recent_template_patterns.pop(0)
                        self._write_stats_snapshot()
                    
                    return (formatted_sentence, words_used)
                # If validation failed, try again with next attempt
                self.rejection_counts['validator'] += 1
                template_id = getattr(template, 'identifier', template.name if hasattr(template, 'name') else str(template))
                self.template_fail_counts[template_id] += 1
                self._write_stats_snapshot()
            
            # All attempts failed validation
            return None
            
        except requests.exceptions.Timeout:
            print(f"\n⚠️  LLM timeout after 30s - Ollama may be overloaded or stuck")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  LLM request failed: {e}")
            return None
        except Exception as e:
            print(f"\n⚠️  Unexpected error in LLM generation: {e}")
            return None
    
    def _format_sentence(self, sentence: str, template: Template) -> str:
        """
        Format sentence with proper capitalization and punctuation.
        
        Args:
            sentence: Raw sentence string
            template: Template used to generate the sentence
            
        Returns:
            Formatted sentence with capitalization and punctuation
        """
        # Strip any existing punctuation marks from both ends
        sentence = sentence.strip('.!?¡¿')
        
        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Add punctuation based on template type
        if template.mood == 'Imp':
            # Imperative - use exclamation marks
            sentence = '¡' + sentence + '!'
        elif any(q in template.glue_words for q in ['¿qué', '¿dónde', '¿cuándo', '¿cómo', '¿quién', '¿por qué']):
            # Question - ensure question marks
            sentence = '¿' + sentence + '?'
        else:
            # Statement - use period
            sentence = sentence + '.'
        
        return sentence

    def _passes_semantic_heuristics(self, sentence: str, template: Template) -> bool:
        """Apply lightweight semantic screening before LLM validation."""
        sentence_lower = sentence.lower()
        
        if 'si ' in sentence_lower:
            if not self._conditional_has_resolution(sentence_lower):
                self._record_rejection('conditional')
                return False
            if ' pero ' in sentence_lower and not self._conditional_has_resolution(sentence_lower, require_resolution=True):
                self._record_rejection('conditional_pero')
                return False
        
        if template.mood == 'Imp':
            if not self._is_valid_imperative(sentence_lower):
                self._record_rejection('imperative')
                return False
        
        if any(fantasy in sentence_lower for fantasy in self.FANTASY_WORDS):
            self._record_rejection('fantasy')
            return False
        
        if any(color in sentence_lower for color in self.COLOR_WORDS):
            if not any(context in sentence_lower for context in self.COLOR_CONTEXT_WORDS):
                self._record_rejection('color_context')
                return False
            if any(abstract in sentence_lower for abstract in self.ABSTRACT_COLOR_WORDS):
                self._record_rejection('color_abstract')
                return False
        
        return True
    
    def _conditional_has_resolution(self, sentence_lower: str, require_resolution: bool = False) -> bool:
        """Heuristically ensure that 'si' clauses include a resolution."""
        if 'si ' not in sentence_lower:
            return True
        
        resolution_markers = [
            ' entonces', ' entonces,', 'entonces ', ' podrá', ' podra', ' podrán', ' podran',
            ' pueden', ' puede', ' podrían', ' podrian', ' permitirá', ' permitira', ' permitirá',
            ' permitira', ' ayudará', ' ayudara', ' ayuda', ' resolverá', ' resolvera',
            ' logrará', ' lograra', ' habrá', ' habra', ' habría', ' habria'
        ]
        if any(marker in sentence_lower for marker in resolution_markers):
            return True
        
        if require_resolution and ' pero ' in sentence_lower:
            return False
        
        parts = sentence_lower.split(',')
        for part in parts[1:]:
            if re.search(r'\b([a-záéíóúñ]+ará|ará|erá|irá|ara|era|ira|rán|ran|rían|rian|ía|ia|an|en)\b', part.strip()):
                return True
        return not require_resolution
    
    def _is_valid_imperative(self, sentence_lower: str) -> bool:
        """Check that imperative sentences start with a plausible command and have objects."""
        cleaned = sentence_lower.strip('¡! ').strip()
        if not cleaned:
            return False
        first_word = cleaned.split()[0]
        if first_word not in self.imperative_forms:
            return False
        
        for word in cleaned.split():
            token = word.strip('.,;:¡!¿?"\'')
            info = self.classifier.get_word_info(token) or self.classifier.get_word_info(token.lower())
            if info.get('pos') in ('NOUN', 'PROPN'):
                return True
        return False

    def _record_rejection(self, reason: str):
        """Track heuristic rejection reasons."""
        self.rejection_counts['heuristic'] += 1
        key = f'heuristic_{reason}'
        self.rejection_counts[key] += 1
        self._write_stats_snapshot()
    
    def generate_sentences(self, verbose: bool = True) -> List[str]:
        """
        Generate sentences to cover all words.
        
        Args:
            verbose: Print progress information
            
        Returns:
            List of generated sentences
        """
        return self._generate_sentences_standard(verbose)
    
    def _generate_sentences_standard(self, verbose: bool = True) -> List[str]:
        """
        Standard sentence generation (faster, more variety).
        
        Args:
            verbose: Print progress information
            
        Returns:
            List of generated sentences
        """
        sentences = self.sentences
        templates = self.template_library.get_templates(self.max_words)
        
        if verbose:
            print(f"\nGenerating sentences (max {self.max_words} words per sentence)...")
            print(f"Total words to cover: {len(self.unused_words)}")
            print(f"Available templates: {len(templates)}")
            if sentences:
                print(f"\nResuming from {len(sentences)} existing sentences...")
                print(f"Words remaining: {len(self.unused_words)}")
            else:
                print(f"\nStarting generation...")
            print("Note: 'rejections: A (heuristic B, validator C)' shows total discarded sentences –")
            print("      heuristics reject implausible sentences before validation,")
            print("      validator rejections come from the LLM coherence check.")
            if self.stats_file:
                print(f"Live stats file: {self.stats_file}")
            sys.stdout.flush()
        
        iteration = 0
        max_iterations = len(self.unused_words) * 2  # Safety limit
        failed_attempts = 0
        max_failed_attempts = 200  # Stop if we fail 200 times in a row
        batch_start_time = time.time()
        last_batch_count = 0
        
        while self.unused_words and iteration < max_iterations:
            iteration += 1
            
            # Show progress immediately
            if verbose and iteration == 1:
                sys.stdout.write(f'\r  Generating sentence 1... 0.0s elapsed')
                sys.stdout.flush()
            
            # Select template - prefer templates that use rare POS tags
            template = self._select_best_template(templates)
            
            if template is None:
                if verbose:
                    print(f"Warning: No suitable template found. Remaining words: {len(self.unused_words)}")
                break
            
            # Generate sentence with LLM (required)
            if not self.llm_validator:
                if verbose:
                    print("\nError: LLM validator is required for sentence generation")
                break
            
            # Show progress before LLM call
            if verbose:
                elapsed = time.time() - batch_start_time
                # Clear line first, then write new message
                sys.stdout.write('\r' + ' ' * 140 + '\r')
                sys.stdout.write(f'  Generating sentence {len(sentences) + 1}... {elapsed:.1f}s elapsed (calling LLM...) | {self._format_rejection_progress()}')
                sys.stdout.flush()
            
            attempt_start = time.time()
            result = self._generate_with_llm(template)
            
            if result:
                sentence, _ = result
                self.last_sentence_duration = time.time() - attempt_start
                self.total_sentence_duration += self.last_sentence_duration
                self.successful_sentences += 1
                sentences.append(sentence)
                failed_attempts = 0  # Reset counter on success
                
                # Save immediately after each sentence
                if self.output_file:
                    self._save_sentences_incremental(sentences)
                    self._save_checkpoint()
                
                if verbose and len(sentences) % 50 == 0:
                    # Calculate batch time
                    batch_time = time.time() - batch_start_time
                    # Clear the timer line if it exists
                    sys.stdout.write('\r' + ' ' * 140 + '\r')
                    sys.stdout.flush()
                    # Print final message with time
                    print(f"  Generated {len(sentences)} sentences, {len(self.unused_words)} words remaining ({batch_time:.1f}s) | {self._format_rejection_progress()}")
                    # Reset for next batch
                    batch_start_time = time.time()
                    last_batch_count = len(sentences)
                elif verbose:
                    # Show live timer with sentence count
                    elapsed = time.time() - batch_start_time
                    # Clear line first, then write new message
                    sys.stdout.write('\r' + ' ' * 140 + '\r')
                    # Warn if taking too long
                    if elapsed > 120:  # 2 minutes
                        sys.stdout.write(f'  Generating sentence {len(sentences) + 1}... {elapsed:.1f}s elapsed (may be stuck - consider Ctrl+C) | {self._format_rejection_progress()}')
                    else:
                        sys.stdout.write(f'  Generating sentence {len(sentences) + 1}... {elapsed:.1f}s elapsed | {self._format_rejection_progress()}')
                    sys.stdout.flush()
            else:
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    if verbose:
                        # Clear timer line
                        sys.stdout.write('\r' + ' ' * 80 + '\r')
                        sys.stdout.flush()
                        print(f"\nStopping: Unable to generate valid sentences after {max_failed_attempts} attempts.")
                        print(f"Remaining words may lack required morphological features or don't fit templates.")
                    break
        
        if verbose:
            print(f"\nGeneration complete!")
            print(f"  Total sentences: {len(sentences)}")
            print(f"  Words covered: {len(self.used_words)}")
            print(f"  Words remaining: {len(self.unused_words)}")
            rejection_summary = self._format_rejection_progress()
            print(f"  {rejection_summary}")
            if self.template_fail_counts:
                print("\nTop template failures:")
                sorted_templates = sorted(self.template_fail_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for template_id, count in sorted_templates:
                    print(f"  - {template_id}: {count}")
            stats_summary = self.get_statistics()
            print(f"  Last sentence time: {stats_summary['last_sentence_duration']:.2f}s")
            print(f"  Avg sentence time : {stats_summary['avg_sentence_duration']:.2f}s")
            
            if self.unused_words:
                print(f"\nUnused words by POS:")
                unused_by_pos = defaultdict(list)
                for word in self.unused_words:
                    info = self.classifier.get_word_info(word)
                    pos = info.get('pos', 'UNKNOWN')
                    unused_by_pos[pos].append(word)
                
                for pos, words in sorted(unused_by_pos.items()):
                    print(f"  {pos}: {len(words)} words")
                    if len(words) <= 5:
                        print(f"    {', '.join(words)}")
        
        return sentences
    
    def _select_best_template(self, templates: List[Template]) -> Optional[Template]:
        """
        Select the best template based on unused words and mood variety.
        
        Args:
            templates: List of available templates
            
        Returns:
            Best template or None
        """
        # Track mood usage for variety
        if not hasattr(self, '_mood_counts'):
            self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        
        # Score each template based on how many unused words it can use
        scored_templates = []
        
        for template in templates:
            score = 0
            can_fill = True
            
            for pos in template.slots:
                unused = self.get_unused_words_by_pos(pos)
                all_pos = self.classifier.get_words_by_pos(pos)
                
                if unused:
                    score += 10  # Bonus for using unused words
                elif not all_pos:
                    can_fill = False
                    break
            
            if can_fill:
                # Add variety bonus for underused moods
                template_mood = template.mood or 'Ind'
                
                # Check if it's a question
                is_question = any(q in template.glue_words for q in ['¿qué', '¿dónde', '¿cuándo', '¿cómo', '¿quién', '¿por qué'])
                if is_question:
                    template_mood = 'Question'
                
                # Boost score for underused moods (encourage variety)
                total_sentences = sum(self._mood_counts.values())
                if total_sentences > 0:
                    mood_ratio = self._mood_counts.get(template_mood, 0) / total_sentences
                    # Boost underused moods (target: 60% Ind, 15% Imp, 15% Sub, 10% Question)
                    target_ratios = {'Ind': 0.60, 'Imp': 0.15, 'Sub': 0.15, 'Question': 0.10}
                    target = target_ratios.get(template_mood, 0.60)
                    if mood_ratio < target:
                        score += 20  # Strong boost for underused moods
                
                pattern = self._template_pattern_signature(template)
                if pattern and self._recent_template_patterns:
                    recent = self._recent_template_patterns[-8:]
                    repetition = recent.count(pattern)
                    if pattern == 'si+pero' and repetition >= 2:
                        score -= 20
                    elif repetition >= 3:
                        score -= 10
                
                scored_templates.append((score, template, template_mood))
        
        if not scored_templates:
            return None
        
        # Sort by score and pick from top candidates
        scored_templates.sort(reverse=True, key=lambda x: x[0])
        
        # Pick randomly from top 5 to add variety
        top_candidates = scored_templates[:min(5, len(scored_templates))]
        selected_score, selected_template, selected_mood = random.choice(top_candidates)
        
        # Track mood usage
        self._mood_counts[selected_mood] = self._mood_counts.get(selected_mood, 0) + 1
        
        return selected_template
    
    def _template_pattern_signature(self, template: Template) -> Optional[str]:
        """Return a descriptor of connector usage for repetition tracking."""
        connectors = []
        glue = [word.lower() for word in template.glue_words]
        if 'si' in glue:
            connectors.append('si')
        if 'pero' in glue:
            connectors.append('pero')
        if 'cuando' in glue:
            connectors.append('cuando')
        if 'porque' in glue:
            connectors.append('porque')
        if not connectors:
            return None
        return '+'.join(sorted(connectors))
    
    def _format_rejection_progress(self) -> str:
        """Human-readable summary of rejection counts for progress output."""
        heuristic = self.rejection_counts.get('heuristic', 0)
        validator = self.rejection_counts.get('validator', 0)
        total = heuristic + validator
        if total == 0:
            return "rejections: 0"
        return f"rejections: {total} (heuristic {heuristic}, validator {validator})"
    
    def get_statistics(self) -> dict:
        """
        Get statistics about word usage and sentence variety.
        
        Returns:
            Dictionary with usage statistics
        """
        total_words = len(self.used_words) + len(self.unused_words)
        
        # Get mood statistics if available
        mood_stats = {}
        if hasattr(self, '_mood_counts'):
            mood_stats = dict(self._mood_counts)
        
        stats = {
            'total_words': total_words,
            'used_words': len(self.used_words),
            'unused_words': len(self.unused_words),
            'coverage_percent': (len(self.used_words) / total_words * 100) if total_words > 0 else 0,
            'avg_word_usage': sum(self.word_usage_count.values()) / len(self.word_usage_count) if self.word_usage_count else 0,
            'max_word_usage': max(self.word_usage_count.values()) if self.word_usage_count else 0,
            'llm_generated': self.llm_generated_count,
            'rejections': dict(self.rejection_counts),
            'template_failures': dict(self.template_fail_counts),
            'last_sentence_duration': self.last_sentence_duration,
            'avg_sentence_duration': (self.total_sentence_duration / self.successful_sentences) if self.successful_sentences else 0.0
        }
        
        # Add mood statistics if available
        if mood_stats:
            stats['mood_counts'] = mood_stats
        
        return stats
