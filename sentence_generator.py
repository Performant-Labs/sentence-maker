"""
Sentence Generator Module
Generates grammatically correct Spanish sentences using LLM with template guidance.
"""
import json
import requests
import hashlib
import math
import os
import random
import re
import time
import sys
import unicodedata
from typing import List, Optional, Tuple
from collections import Counter, defaultdict, deque

from spellchecker import SpellChecker
from word_classifier import WordClassifier
from sentence_templates import TemplateLibrary, Template


class SentenceGenerator:
    """Generates Spanish sentences to cover all words in a vocabulary."""

    # ============================================================================
    # CONFIGURATION: Generation Parameters
    # ============================================================================

    # LLM Word Pool Size
    # Number of words per POS category provided to the LLM for sentence generation.
    # Higher values give the LLM more semantic options to create coherent sentences,
    # but may increase prompt size and processing time.
    # - Original: 20 words (limited semantic variety)
    # - Previous: 75 words (good balance)
    # - Current: 100 words (maximum variety with 5 retry attempts)
    # - Maximum recommended: 100 words (diminishing returns beyond this)
    LLM_WORD_POOL_SIZE = 100

    # LLM Generation Retry Attempts
    # Number of times to retry LLM generation before falling back to simple sentences.
    # Higher values reduce fallback usage but increase generation time.
    # - Original: 2 attempts (too few with strict validation)
    # - Current: 5 attempts (good balance)
    # - Maximum recommended: 10 attempts (diminishing returns)
    LLM_MAX_ATTEMPTS = 5

    # ============================================================================
    # CONFIGURATION: Semantic Validation
    # ============================================================================

    # Color Word Validation
    # Ensures colors only appear with physical objects, not abstract concepts.
    # Prevents nonsensical combinations like "red emotion" or "blue time".
    COLOR_WORDS = {
        'rojo', 'rojos', 'roja', 'rojas',
        'blanco', 'blancos', 'blanca', 'blancas',
        'negro', 'negros', 'negra', 'negras',
        'verde', 'verdes', 'amarillo', 'amarilla', 'amarillos', 'amarillas',
        'azul', 'azules', 'morado', 'morado', 'morada', 'moradas',
        'gris', 'grises'
    }
    # Physical objects that can have colors
    COLOR_CONTEXT_WORDS = {
        'casa', 'hogar', 'pared', 'paredes', 'habitacion', 'habitación', 'puerta',
        'ventana', 'techo', 'sala', 'sillón', 'sillon', 'auto', 'automóvil', 'automovil',
        'vehículo', 'vehiculo', 'coche', 'camioneta', 'camión', 'camion', 'camisa', 'camiseta',
        'vestido', 'falda', 'zapato', 'zapatos', 'ropa', 'sombrero', 'mesa', 'silla', 'sofá',
        'sofa', 'carro', 'bicicleta', 'flor', 'flores', 'árbol', 'arbol', 'fruta', 'frutas',
        'manzana', 'sandía', 'sandia', 'cama', 'colcha', 'cocina', 'plato', 'comida',
        'bebida', 'taza', 'cuadro', 'decoración', 'decoracion'
    }
    # Abstract concepts that should NOT be combined with colors
    ABSTRACT_COLOR_WORDS = {
        'sentimiento', 'sentimientos', 'emocion', 'emoción', 'emociones', 'temporada',
        'sensación', 'sensacion', 'psicologia', 'psicología', 'nombramiento', 'nombramientos',
        'signos', 'reglamento', 'reglamentos', 'salud', 'tiempo', 'rutina'
    }
    # ============================================================================
    # CONFIGURATION: Question Words
    # ============================================================================

    # Spanish interrogative words for question detection
    QUESTION_WORDS = {
        'que', 'qué', 'quien', 'quién', 'quienes', 'quiénes', 'donde', 'dónde',
        'cuando', 'cuándo', 'como', 'cómo', 'cual', 'cuál', 'cuales', 'cuáles',
        'cuanto', 'cuánto', 'cuanta', 'cuánta', 'cuantos', 'cuántos', 'cuantas', 'cuántas'
    }
    # ============================================================================
    # CONFIGURATION: Topic Seeding
    # ============================================================================

    # Seed topics with associated keywords for smart topic selection.
    # When available words match a topic's keywords (≥2 matches), the LLM is
    # guided to generate sentences about that topic for better coherence.
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
    # ============================================================================
    # CONFIGURATION: Performance & Statistics
    # ============================================================================

    # Duration buckets for sentence generation timing histogram
    DURATION_BUCKETS = ["<4s", "4-8s", "8-12s", "12-16s", "16-20s", ">=20s"]

    # Pattern for extracting Spanish words (includes accented characters)
    WORD_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

    # ============================================================================
    # CONFIGURATION: Repetition Detection
    # ============================================================================

    # Stopwords to ignore when checking for repetitive phrases.
    # These common function words don't contribute to semantic repetition.
    REPETITION_STOPWORDS = {
        'de', 'la', 'las', 'el', 'los', 'y', 'o', 'u', 'que', 'se', 'a', 'en',
        'por', 'para', 'del', 'al', 'con', 'sin', 'su', 'sus', 'mi', 'mis',
        'tu', 'tus', 'su', 'sus', 'este', 'esta', 'estos', 'estas', 'eso', 'esa',
        'eso', 'esas', 'un', 'una', 'unos', 'unas', 'lo'
    }
    # Sentence start tracking - prevents repetitive sentence beginnings
    MAX_START_MEMORY = 60          # Maximum number of recent sentence starts to remember
    START_REPEAT_WINDOW = 50       # Check for duplicates within this window

    # ============================================================================
    # CONFIGURATION: Similarity Detection
    # ============================================================================

    # Prevents generating sentences too similar to recent ones.
    # Uses lexical, structural, and semantic similarity scoring.
    MAX_SIMILARITY_HISTORY = 400           # Number of recent sentences to compare against
    SIMILARITY_THRESHOLD = 0.60            # Reject if similarity score exceeds this
    SIMILARITY_THRESHOLD_FLOOR = 0.35      # Minimum threshold (adaptive)
    SIMILARITY_ADAPT_MIN_SCORES = 80       # Minimum scores before adapting threshold
    SIMILARITY_ADAPT_MARGIN = 0.02         # Threshold adjustment increment
    SIMILARITY_WEIGHTS = (0.4, 0.2, 0.4)   # (lexical, structural, semantic) weights
    SENTENCE_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # For semantic similarity

    def __init__(self, classifier: WordClassifier, max_words: int = 10, output_file: str = None, llm_validator=None, generation_model: str = None, generation_provider: str = None, skip_llm_validation: bool = False, max_sentences: int = 0, enable_spellcheck: bool = False, run_id: int = 0, min_words: int = 5):
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
        # Allow up to 20% over the requested max as a hard cap
        self.max_words_allowed = math.ceil(max_words * 1.2)
        self.output_file = output_file
        self.llm_validator = llm_validator
        self.generation_model = generation_model or (llm_validator.model if llm_validator else None)
        self.generation_provider = generation_provider or (llm_validator.provider if llm_validator else None)
        self.skip_llm_validation = skip_llm_validation
        self.max_words_min = min_words
        self.template_library = TemplateLibrary()
        self.run_id = run_id
        self._template_pos_set = set()
        for template in self.template_library.get_all_templates():
            for slot in template.slots:
                self._template_pos_set.add(slot)
        self._fallback_template = Template(name="FALLBACK_DIRECT", slots=[], glue_words=[], pattern="")
        self.used_words = set()
        self.unused_words = set()
        self.word_usage_count = defaultdict(int)
        self.llm_generated_count = 0
        self.recent_sentence_starts = []  # Track recent sentence beginnings to avoid repetition
        self._normalized_sentence_starts = []
        self.sentences = []
        self.wordlist_checksum = None
        self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._stalled_words = defaultdict(int)
        self._recent_template_patterns = []
        self.imperative_forms = set()
        self.min_word_usage = 1
        self.rejection_counts = defaultdict(int)
        self.stats_file = None
        self.template_fail_counts = defaultdict(int)
        self.template_success_counts = defaultdict(int)
        self.last_sentence_duration = 0.0
        self.first_sentence_duration = None
        self.total_sentence_duration = 0.0
        self.successful_sentences = 0
        self.duration_histogram = defaultdict(int)
        self.start_time = time.time()
        self.max_sentences = max_sentences
        self._unused_pos_counts = defaultdict(int)
        self.spellcheck_enabled = enable_spellcheck
        self.spell_checker = SpellChecker(language='es') if enable_spellcheck else None
        self._spellchecker_whitelist = set()
        self.known_words = set()
        self.spellcheck_duration = 0.0
        self.spellcheck_unknown_counts = defaultdict(int)
        self.last_run_aborted = False
        self._sentence_encoder = None
        self._similarity_records = []
        self._similarity_threshold = self.SIMILARITY_THRESHOLD
        self._similarity_score_history = deque(maxlen=200)
        self._similarity_guard_stats = {
            'combined_rejects': 0,
            'max_combined_score': 0.0,
            'last_combined_score': 0.0,
            'last_semantic_score': 0.0,
            'last_lexical_score': 0.0,
            'last_structural_score': 0.0,
            'cache_rebuilds': 0,
            'cache_size': 0,
            'current_threshold': self.SIMILARITY_THRESHOLD,
            'threshold_adjustments': 0
        }
        self._start_bigrams = []
        self._rejection_log = []
        self._rejection_log_path = None

    def initialize_word_tracking(self, all_words: List[str], resume_state: Optional[dict] = None):
        """
        Initialize tracking for which words have been used.

        Args:
            all_words: Complete list of words to track
            resume_state: Optional checkpoint data to resume progress
        """
        self.wordlist_checksum = self.compute_wordlist_checksum(all_words)
        self._build_imperative_forms()
        if self.spellcheck_enabled:
            self._prepare_spellchecker(all_words)
        if self.output_file:
            self.stats_file = f"{self.output_file}.stats.txt"
        else:
            self.stats_file = "sentencemaker_stats.txt"
        if self.output_file:
            self._rejection_log_path = f"{self.output_file}.rejects.log"
        else:
            self._rejection_log_path = "sentencemaker_rejects.log"

        if resume_state:
            self._restore_from_checkpoint(resume_state, all_words)
        else:
            self._start_fresh(all_words)
        self._rebuild_similarity_caches()
        self._write_stats_snapshot()
        # Reset rejection log file at start
        if self._rejection_log_path:
            try:
                with open(self._rejection_log_path, 'w', encoding='utf-8'):
                    pass
            except Exception:
                pass

    def _start_fresh(self, all_words: List[str]):
        """Reset all tracking structures for a new run."""
        self.unused_words = set(all_words)
        self.used_words = set()
        self.word_usage_count = defaultdict(int)
        self.sentences = []
        self.llm_generated_count = 0
        self.recent_sentence_starts = []
        self._normalized_sentence_starts = []
        self._stalled_words = defaultdict(int)
        self._start_bigrams = []
        self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._recent_template_patterns = []
        self.rejection_counts = defaultdict(int)
        self.last_sentence_duration = 0.0
        self.first_sentence_duration = None
        self.total_sentence_duration = 0.0
        self.successful_sentences = 0
        self.duration_histogram = defaultdict(int)
        self.spellcheck_duration = 0.0
        self.spellcheck_unknown_counts = defaultdict(int)
        self.last_run_aborted = False
        self._similarity_records = []
        self._similarity_threshold = self.SIMILARITY_THRESHOLD
        self._similarity_score_history.clear()
        self._similarity_guard_stats.update({
            'combined_rejects': 0,
            'max_combined_score': 0.0,
            'last_combined_score': 0.0,
            'last_semantic_score': 0.0,
            'last_lexical_score': 0.0,
            'last_structural_score': 0.0,
            'cache_rebuilds': self._similarity_guard_stats.get('cache_rebuilds', 0),
            'cache_size': 0,
            'current_threshold': self._similarity_threshold,
            'threshold_adjustments': 0
        })
        self._recompute_unused_pos_counts()

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
        self._normalized_sentence_starts = [
            self._normalize_text_for_similarity(start) for start in self.recent_sentence_starts if start
        ]
        self._start_bigrams = state.get('start_bigrams', [])
        mood_counts = state.get('mood_counts')
        if mood_counts:
            self._mood_counts = mood_counts
        else:
            self._mood_counts = {'Ind': 0, 'Imp': 0, 'Sub': 0, 'Question': 0}
        self._recent_template_patterns = state.get('template_patterns', [])
        self.rejection_counts = defaultdict(int, state.get('rejection_counts', {}))
        self.template_fail_counts = defaultdict(int, state.get('template_failures', {}))
        self.template_success_counts = defaultdict(int, state.get('template_successes', {}))
        self.last_sentence_duration = state.get('last_sentence_duration', 0.0)
        self.total_sentence_duration = state.get('total_sentence_duration', 0.0)
        self.successful_sentences = state.get('successful_sentences', len(self.sentences))
        self.first_sentence_duration = state.get('first_sentence_duration')
        self.spellcheck_duration = state.get('spellcheck_duration', 0.0)
        self.spellcheck_unknown_counts = defaultdict(int, state.get('spellcheck_unknown_counts', {}))
        self.last_run_aborted = False
        self.duration_histogram = defaultdict(int, state.get('duration_histogram', {}))
        self.max_sentences = state.get('max_sentences', self.max_sentences)
        guard_stats = state.get('similarity_guard_stats')
        if guard_stats:
            self._similarity_guard_stats.update(guard_stats)
        self._similarity_threshold = state.get('similarity_threshold', self.SIMILARITY_THRESHOLD)
        self._similarity_guard_stats['current_threshold'] = self._similarity_threshold
        self._similarity_score_history.clear()

        if self.output_file:
            # Ensure checkpoint output exists and reflects current sentences
            self._save_sentences_incremental(self.sentences)
        self._recompute_unused_pos_counts()

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

    def _prepare_spellchecker(self, words: List[str]):
        """Load vocabulary into the spell checker for custom coverage."""
        if not self.spell_checker:
            return
        tokens = set()
        sources = [words]
        if hasattr(self.classifier, 'word_info'):
            sources.append(list(self.classifier.word_info.keys()))
        for source in sources:
            for word in source:
                if not word:
                    continue
                for token in self.WORD_PATTERN.findall(word):
                    normalized = token.lower()
                    if normalized:
                        tokens.add(normalized)
        if tokens and self.spell_checker:
            self.spell_checker.word_frequency.load_words(tokens)
            self._spellchecker_whitelist.update(tokens)
        self.known_words = set(self._spellchecker_whitelist)

    def _recompute_unused_pos_counts(self):
        """Recalculate counts of unused words per POS."""
        self._unused_pos_counts = defaultdict(int)
        if not self.classifier:
            return
        for word in self.unused_words:
            info = self.classifier.get_word_info(word)
            pos = info.get('pos')
            if pos:
                self._unused_pos_counts[pos] += 1

    def _rebuild_similarity_caches(self):
        """Rebuild semantic/lexical caches from the sentences on disk."""
        # Keep normalized starts aligned with raw starts
        if len(self._normalized_sentence_starts) != len(self.recent_sentence_starts):
            self._normalized_sentence_starts = [
                self._normalize_text_for_similarity(start) for start in self.recent_sentence_starts if start
            ]
        else:
            self._normalized_sentence_starts = [
                self._normalize_text_for_similarity(start) if start else None
                for start in self.recent_sentence_starts
            ]
        if len(self._start_bigrams) != len(self.recent_sentence_starts):
            self._start_bigrams = []
            for start in self.recent_sentence_starts:
                tokens = start.split() if start else []
                if len(tokens) >= 2:
                    self._start_bigrams.append(f"{tokens[0].lower()} {tokens[1].lower()}")
                elif tokens:
                    self._start_bigrams.append(tokens[0].lower())
                else:
                    self._start_bigrams.append("")
        self._similarity_records = []
        sentences_to_index = self.sentences[-self.MAX_SIMILARITY_HISTORY:]
        for sentence in sentences_to_index:
            try:
                features = self._build_similarity_features(sentence)
            except RuntimeError:
                raise
            self._register_sentence_similarity(features)
        self._similarity_guard_stats['cache_rebuilds'] = self._similarity_guard_stats.get('cache_rebuilds', 0) + 1
        self._similarity_guard_stats['cache_size'] = len(self._similarity_records)
        self._similarity_guard_stats['current_threshold'] = self._similarity_threshold

    def _ensure_sentence_encoder(self):
        """Lazily load the sentence embedding model."""
        if self._sentence_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for repetition prevention. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._sentence_encoder = SentenceTransformer(self.SENTENCE_EMBEDDING_MODEL)
        return self._sentence_encoder

    def _normalize_text_for_similarity(self, text: Optional[str]) -> str:
        """Normalize text for lexical comparisons (lowercase, strip accents/punctuation)."""
        if not text:
            return ""
        text = text.replace('¡', ' ').replace('¿', ' ')
        normalized = unicodedata.normalize('NFD', text)
        cleaned = []
        for char in normalized.lower():
            if unicodedata.category(char) == 'Mn':
                continue
            cleaned.append(char if char.isalnum() or char.isspace() else ' ')
        return ' '.join(''.join(cleaned).split())

    def _extract_sentence_start(self, sentence: str) -> Tuple[Optional[str], Optional[str]]:
        """Return the raw and normalized start of a sentence for repetition tracking."""
        words = sentence.split()
        sentence_start = None
        for length in (5, 4, 3):
            if len(words) >= length:
                sentence_start = ' '.join(words[:length])
                break
        if sentence_start is None and len(words) >= 2:
            sentence_start = ' '.join(words[:2])
        elif sentence_start is None and words:
            sentence_start = words[0]
        normalized_start = self._normalize_text_for_similarity(sentence_start) if sentence_start else None
        return sentence_start, normalized_start

    def _compute_start_bigram(self, sentence: str) -> str:
        """Extract the first two words (lowercase, normalized) for repetition tracking."""
        words = sentence.strip('¡¿').split()
        if len(words) >= 2:
            return f"{words[0].lower()} {words[1].lower()}"
        if words:
            return words[0].lower()
        return ""

    def _is_recent_start(self, raw_start: Optional[str], normalized_start: Optional[str], start_bigram: Optional[str]) -> bool:
        """Check if a start phrase has been used recently (raw or normalized)."""
        if not raw_start:
            return False
        if self.recent_sentence_starts and raw_start:
            recent_raw = self.recent_sentence_starts[-self.START_REPEAT_WINDOW:]
            if raw_start in recent_raw:
                return True
        if normalized_start and self._normalized_sentence_starts:
            recent_norm = [value for value in self._normalized_sentence_starts[-self.START_REPEAT_WINDOW:] if value]
            if normalized_start in recent_norm:
                return True
        if start_bigram:
            recent_bigrams = [value for value in self._start_bigrams[-self.START_REPEAT_WINDOW:] if value]
            if start_bigram in recent_bigrams:
                return True
        return False

    def _append_sentence_start(self, raw_start: Optional[str], normalized_start: Optional[str], start_bigram: Optional[str]):
        """Record the raw and normalized sentence starts with proper trimming."""
        if not raw_start:
            return
        normalized = normalized_start or self._normalize_text_for_similarity(raw_start)
        self.recent_sentence_starts.append(raw_start)
        self._normalized_sentence_starts.append(normalized)
        self._start_bigrams.append(start_bigram or "")
        self._trim_sentence_starts()

    def _trim_sentence_starts(self):
        """Keep the stored sentence starts within the configured memory window."""
        while len(self.recent_sentence_starts) > self.MAX_START_MEMORY:
            self.recent_sentence_starts.pop(0)
            if self._normalized_sentence_starts:
                self._normalized_sentence_starts.pop(0)
            if self._start_bigrams:
                self._start_bigrams.pop(0)
        while len(self._normalized_sentence_starts) > len(self.recent_sentence_starts):
            self._normalized_sentence_starts.pop(0)
        while len(self._start_bigrams) > len(self.recent_sentence_starts):
            self._start_bigrams.pop(0)

    def _build_similarity_features(
        self,
        sentence: str,
        start_data: Optional[Tuple[Optional[str], Optional[str], str]] = None
    ) -> dict:
        """Create lexical, structural, and semantic representations of a sentence."""
        normalized_sentence = self._normalize_text_for_similarity(sentence)
        word_counter, word_total = self._collect_word_ngrams(normalized_sentence)
        char_counter, char_total = self._collect_char_ngrams(normalized_sentence)
        pos_counter, pos_total = self._collect_pos_ngrams(sentence)
        embedding_vector, embedding_norm = self._compute_sentence_embedding(sentence)
        start_raw = start_normalized = start_bigram = None
        if start_data:
            start_raw, start_normalized, start_bigram = start_data
        else:
            start_raw, start_normalized = self._extract_sentence_start(sentence)
            start_bigram = self._compute_start_bigram(sentence)
        return {
            'sentence': sentence,
            'normalized': normalized_sentence,
            'start_raw': start_raw,
            'start_normalized': start_normalized,
            'start_bigram': start_bigram,
            'word_counter': word_counter,
            'word_len': word_total,
            'char_counter': char_counter,
            'char_len': char_total,
            'pos_counter': pos_counter,
            'pos_len': pos_total,
            'embedding': embedding_vector,
            'embedding_norm': embedding_norm
        }

    def _collect_word_ngrams(self, normalized_sentence: str) -> Tuple[Counter, int]:
        """Return Counter of word 1-2 grams and their total count."""
        tokens = normalized_sentence.split()
        grams: List[str] = []
        for size in (1, 2):
            if len(tokens) < size:
                continue
            for idx in range(len(tokens) - size + 1):
                grams.append(' '.join(tokens[idx:idx + size]))
        counter = Counter(grams)
        return counter, len(grams)

    def _collect_char_ngrams(self, normalized_sentence: str) -> Tuple[Counter, int]:
        """Return Counter of character n-grams (3-5) with word-boundary style padding."""
        if not normalized_sentence:
            return Counter(), 0
        grams: List[str] = []
        padded = f" {normalized_sentence} "
        for size in range(3, 6):
            if len(padded) < size:
                continue
            for idx in range(len(padded) - size + 1):
                gram = padded[idx:idx + size]
                if gram.strip():
                    grams.append(gram)
        counter = Counter(grams)
        return counter, len(grams)

    def _collect_pos_ngrams(self, sentence: str) -> Tuple[Counter, int]:
        """Return Counter of POS n-grams (1-3) using the classifier's spaCy pipeline."""
        if not self.classifier or not hasattr(self.classifier, 'nlp'):
            return Counter(), 0
        tags = []
        doc = self.classifier.nlp(sentence)
        for token in doc:
            if not token.is_space:
                tags.append(token.pos_)
        grams: List[str] = []
        for size in (1, 2, 3):
            if len(tags) < size:
                continue
            for idx in range(len(tags) - size + 1):
                grams.append('_'.join(tags[idx:idx + size]))
        counter = Counter(grams)
        return counter, len(grams)

    def _compute_sentence_embedding(self, sentence: str) -> Tuple[List[float], float]:
        """Encode a sentence using SentenceTransformer and return vector + norm."""
        encoder = self._ensure_sentence_encoder()
        embedding = encoder.encode(sentence, convert_to_numpy=True, normalize_embeddings=True)
        if hasattr(embedding, 'tolist'):
            vector = embedding.tolist()
        else:
            vector = list(embedding)
        norm = math.sqrt(sum(value * value for value in vector)) or 1e-12
        return vector, norm

    def _register_sentence_similarity(self, features: dict):
        """Store similarity features for a newly accepted sentence."""
        if not features:
            return
        record = {
            'sentence': features['sentence'],
            'normalized': features['normalized'],
            'start_raw': features['start_raw'],
            'start_normalized': features['start_normalized'],
            'start_bigram': features.get('start_bigram', ''),
            'word_counter': features['word_counter'],
            'word_len': features['word_len'],
            'char_counter': features['char_counter'],
            'char_len': features['char_len'],
            'pos_counter': features['pos_counter'],
            'pos_len': features['pos_len'],
            'embedding': features['embedding'],
            'embedding_norm': features['embedding_norm']
        }
        self._similarity_records.append(record)
        self._trim_similarity_history()
        self._similarity_guard_stats['cache_size'] = len(self._similarity_records)

    def _trim_similarity_history(self):
        """Bound the similarity cache to a fixed history length."""
        while len(self._similarity_records) > self.MAX_SIMILARITY_HISTORY:
            self._similarity_records.pop(0)

    def _shorten_sentence_with_llm(self, sentence: str) -> Optional[str]:
        """Rewrite a sentence under the max word limit while keeping meaning."""
        if not self.llm_validator or not self.generation_provider:
            return None
        prompt = f"""Reescribe esta oración en español con NO MÁS de {self.max_words} palabras.
Mantén el significado, arregla concordancia y gramática si es necesario.
Devuelve solo la oración reescrita, sin explicaciones.

Oración: {sentence}
Reescrita:"""
        try:
            if self.generation_provider == 'openai' or (self.llm_validator and self.llm_validator.provider == 'openai'):
                return self.llm_validator.generate_openai(prompt, temperature=0.2, max_tokens=24, model_override=self.generation_model)
            if self.generation_provider == 'anthropic' or (self.llm_validator and self.llm_validator.provider == 'anthropic'):
                return self.llm_validator.generate_anthropic(prompt, temperature=0.2, max_tokens=48, model_override=self.generation_model)
        except Exception:
            return None
        return None

    @staticmethod
    def _cosine_similarity(vec_a: List[float], norm_a: float, vec_b: List[float], norm_b: float) -> float:
        """Compute cosine similarity between two dense vectors."""
        if not vec_a or not vec_b or not norm_a or not norm_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        if not dot:
            return 0.0
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _overlap_ratio(counter_a: Counter, counter_b: Counter) -> float:
        """Return overlap ratio between two Counters."""
        if not counter_a or not counter_b:
            return 0.0
        intersection = sum(min(counter_a[token], counter_b[token]) for token in counter_a.keys() & counter_b.keys())
        union = sum(max(counter_a.get(token, 0), counter_b.get(token, 0)) for token in set(counter_a) | set(counter_b))
        if not union:
            return 0.0
        return intersection / union

    def _is_repetitive_via_similarity(self, features: dict) -> Tuple[bool, dict]:
        """Compare against cached sentences using weighted lexical/structural/semantic similarity."""
        if not self._similarity_records:
            return False, {'combined': 0.0, 'lexical': 0.0, 'structural': 0.0, 'semantic': 0.0}
        if not features:
            return False, {'combined': 0.0, 'lexical': 0.0, 'structural': 0.0, 'semantic': 0.0}
        total_docs = len(self._similarity_records)
        word_counter = features.get('word_counter') or Counter()
        char_counter = features.get('char_counter') or Counter()
        pos_counter = features.get('pos_counter') or Counter()
        best_scores = {'combined': 0.0, 'lexical': 0.0, 'structural': 0.0, 'semantic': 0.0}
        for record in reversed(self._similarity_records):
            lexical_word = self._overlap_ratio(word_counter, record['word_counter'])
            lexical_char = self._overlap_ratio(char_counter, record['char_counter'])
            lexical_components = [score for score in (lexical_word, lexical_char) if score]
            lexical_score = sum(lexical_components) / len(lexical_components) if lexical_components else 0.0
            structural_score = self._overlap_ratio(pos_counter, record['pos_counter'])
            semantic_score = self._cosine_similarity(
                features['embedding'],
                features['embedding_norm'],
                record['embedding'],
                record['embedding_norm']
            )
            combined = (
                self.SIMILARITY_WEIGHTS[0] * lexical_score +
                self.SIMILARITY_WEIGHTS[1] * structural_score +
                self.SIMILARITY_WEIGHTS[2] * semantic_score
            )
            if combined > best_scores['combined']:
                best_scores = {
                    'combined': combined,
                    'lexical': lexical_score,
                    'structural': structural_score,
                    'semantic': semantic_score
                }
            if combined >= self._similarity_threshold:
                return True, best_scores
        return False, best_scores

    def _update_similarity_stats(self, detail: dict, rejected: bool):
        """Record instrumentation for similarity guard behavior."""
        if not detail:
            return
        self._similarity_guard_stats['last_combined_score'] = detail.get('combined', 0.0)
        self._similarity_guard_stats['last_lexical_score'] = detail.get('lexical', 0.0)
        self._similarity_guard_stats['last_structural_score'] = detail.get('structural', 0.0)
        self._similarity_guard_stats['last_semantic_score'] = detail.get('semantic', 0.0)
        self._similarity_guard_stats['current_threshold'] = self._similarity_threshold
        if rejected:
            self._similarity_guard_stats['combined_rejects'] = self._similarity_guard_stats.get('combined_rejects', 0) + 1
            self._similarity_guard_stats['last_rejection_score'] = detail.get('combined', 0.0)
        self._similarity_guard_stats['max_combined_score'] = max(
            self._similarity_guard_stats.get('max_combined_score', 0.0),
            detail.get('combined', 0.0)
        )
        self._similarity_guard_stats['cache_size'] = len(self._similarity_records)

    def _record_similarity_score(self, combined_score: float):
        """Track similarity scores and adapt the threshold if no rejects occur."""
        if combined_score is None or combined_score <= 0.0:
            return
        self._similarity_score_history.append(combined_score)
        self._similarity_guard_stats['score_observations'] = len(self._similarity_score_history)
        self._similarity_guard_stats['current_threshold'] = self._similarity_threshold
        history_len = len(self._similarity_score_history)
        history_full = history_len == self._similarity_score_history.maxlen
        has_rejects = self._similarity_guard_stats.get('combined_rejects', 0) > 0
        if (history_len < self.SIMILARITY_ADAPT_MIN_SCORES and not history_full) or has_rejects:
            return
        observed_max = max(self._similarity_score_history)
        if observed_max <= 0.0:
            return
        target_threshold = max(
            self.SIMILARITY_THRESHOLD_FLOOR,
            min(self.SIMILARITY_THRESHOLD, observed_max - self.SIMILARITY_ADAPT_MARGIN)
        )
        if target_threshold < self._similarity_threshold - 0.01:
            self._similarity_threshold = target_threshold
        self._similarity_guard_stats['current_threshold'] = self._similarity_threshold
        self._similarity_guard_stats['threshold_adjustments'] = self._similarity_guard_stats.get('threshold_adjustments', 0) + 1
        self._similarity_score_history.clear()
        self._rejection_log = []
        if self.output_file:
            self._rejection_log_path = f"{self.output_file}.rejects.log"
        else:
            self._rejection_log_path = "sentencemaker_rejects.log"

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
            'recent_sentence_starts': self.recent_sentence_starts[-self.MAX_START_MEMORY:],
            'mood_counts': self._mood_counts,
            'wordlist_checksum': self.wordlist_checksum,
            'max_words': self.max_words,
            'template_patterns': self._recent_template_patterns[-40:],
            'rejection_counts': dict(self.rejection_counts),
            'template_failures': dict(self.template_fail_counts),
            'template_successes': dict(self.template_success_counts),
            'last_sentence_duration': self.last_sentence_duration,
            'first_sentence_duration': self.first_sentence_duration,
            'total_sentence_duration': self.total_sentence_duration,
            'spellcheck_duration': self.spellcheck_duration,
            'spellcheck_unknown_counts': dict(self.spellcheck_unknown_counts),
            'successful_sentences': self.successful_sentences,
            'duration_histogram': dict(self.duration_histogram),
            'timestamp': time.time(),
            'start_time': self.start_time,
            'max_sentences': self.max_sentences,
            'similarity_guard_stats': dict(self._similarity_guard_stats),
            'similarity_threshold': self._similarity_threshold,
            'start_bigrams': self._start_bigrams[-self.MAX_START_MEMORY:]
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Ignore checkpoint errors

    def _get_stats_snapshot(self) -> str:
        """Return a human-readable snapshot of current stats."""
        stats = self.get_statistics()
        total_rejections = self.rejection_counts.get('heuristic', 0) + self.rejection_counts.get('validator', 0)
        spellcheck_total_ms = stats.get('spellcheck_duration', 0.0) * 1000.0
        spellcheck_avg_ms = (spellcheck_total_ms / len(self.sentences)) if self.sentences else 0.0

        lines = [
            "SentenceMaker Stats Snapshot",
            "=============================",
            f"Run version: {self.run_id}",
            self._format_three_columns(
                ("Sentences generated", str(len(self.sentences))),
                ("Total rejections", str(total_rejections)),
                ("Heuristic rejects", str(self.rejection_counts.get('heuristic', 0)))
            ),
            self._format_three_columns(
                ("Words used", str(stats['used_words'])),
                ("Words remaining", str(stats['unused_words'])),
                ("Validator rejects", str(self.rejection_counts.get('validator', 0)))
            ),
            self._format_three_columns(
                ("Coverage", f"{stats['coverage_percent']:.2f}%"),
                ("Avg sentence time", f"{stats['avg_sentence_duration']:.2f}s"),
                ("Last sentence time", f"{stats['last_sentence_duration']:.2f}s")
            ),
            self._format_three_columns(
                ("Spell check total", f"{spellcheck_total_ms:.2f} ms" if self.spellcheck_enabled else "disabled"),
                ("Spell check avg", f"{spellcheck_avg_ms:.2f} ms" if self.spellcheck_enabled else "-"),
                ("Elapsed time", self._format_elapsed_time(time.time() - self.start_time))
            ),
            self._format_two_columns(
                "Max sentences",
                str(self.max_sentences) if self.max_sentences else "All words"
            ),
            self._format_two_columns(
                "Words cap (target/allowed)",
                f"{self.max_words}/{self.max_words_allowed}"
            ),
            f"LLM generation model: {self.generation_model or 'N/A'} ({self.generation_provider or 'unknown'})",
            f"LLM validation model: {getattr(self.llm_validator, 'model', 'N/A')} ({getattr(self.llm_validator, 'provider', 'unknown')})",
            ""
        ]
        rejections = stats.get('rejections', {})
        detail_keys = sorted(k for k in rejections if k.startswith('heuristic_'))
        if detail_keys:
            lines.append("Heuristic rejections (breakdown):")
            for key in detail_keys:
                label = key.replace('heuristic_', '').replace('_', ' ')
                lines.append(f"  {label:<30}{rejections[key]:>6}")
            lines.append("")
        if self.duration_histogram:
            lines.append("")
            lines.append("Sentence duration histogram:")
            for bucket in self.DURATION_BUCKETS:
                if bucket in self.duration_histogram:
                    lines.append(f"  - {bucket}: {self.duration_histogram[bucket]}")
        spell_unknowns = stats.get('spellcheck_unknowns', [])
        if spell_unknowns:
            lines.append("")
            lines.append("Top spell-check rejects:")
            for word, count in spell_unknowns:
                lines.append(f"  {word:<20}{count:>6}")
        guard_stats = stats.get('similarity_guard')
        if guard_stats:
            lines.append("")
            lines.append("Similarity guard:")
            threshold_display = guard_stats.get('current_threshold', self._similarity_threshold)
            lines.append(
                f"  cache size: {guard_stats.get('cache_size', 0)} "
                f"(threshold={threshold_display:.2f})"
            )
            lines.append(
                "  last scores (L/S/Se): "
                f"{guard_stats.get('last_lexical_score', 0.0):.3f}/"
                f"{guard_stats.get('last_structural_score', 0.0):.3f}/"
                f"{guard_stats.get('last_semantic_score', 0.0):.3f}"
            )
            lines.append(
                f"  last combined: {guard_stats.get('last_combined_score', 0.0):.3f} "
                f"max: {guard_stats.get('max_combined_score', 0.0):.3f} "
                f"rejects: {guard_stats.get('combined_rejects', 0)}"
            )
            adjustments = guard_stats.get('threshold_adjustments', 0)
            if adjustments:
                lines.append(f"  threshold adjustments: {adjustments}")
        if self.template_fail_counts:
            lines.append("")
            lines.append("Top template failures:")
            sorted_templates = sorted(self.template_fail_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for template_id, count in sorted_templates:
                lines.append(f"  - {template_id}: {count}")
        if self.template_success_counts:
            lines.append("")
            lines.append("Top template selections:")
            sorted_success = sorted(self.template_success_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for template_id, count in sorted_success:
                lines.append(f"  - {template_id}: {count}")
        lines.append("")
        lines.append("Updated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        return "\n".join(lines)

    def _format_two_columns(self, left_label: str, left_value: str, right_label: Optional[str] = None, right_value: Optional[str] = None) -> str:
        left = f"{left_label:<28}{left_value:>12}"
        if right_label and right_value is not None:
            right = f"{right_label:<28}{right_value:>12}"
            return f"{left}    {right}"
        return left

    def _format_three_columns(self, first: tuple[str, str], second: tuple[str, str], third: Optional[tuple[str, str]] = None) -> str:
        col1 = f"{first[0]:<20}{first[1]:>12}"
        col2 = f"{second[0]:<20}{second[1]:>12}"
        if third:
            col3 = f"{third[0]:<20}{third[1]:>12}"
            return f"{col1}    {col2}    {col3}"
        return f"{col1}    {col2}"

    def _format_elapsed_time(self, seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

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

    def _log_debug(self, message: str):
        """Append debug info to the rejection log for troubleshooting."""
        if not self._rejection_log_path:
            return
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._rejection_log_path, 'a', encoding='utf-8') as f:
                f.write(f"[DEBUG {ts}] {message}\n")
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
            info = self.classifier.get_word_info(word)
            pos = info.get('pos') if info else None
            if pos:
                current = self._unused_pos_counts.get(pos, 0)
                self._unused_pos_counts[pos] = max(0, current - 1)
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

    def _select_support_word(self, pos: str) -> Optional[str]:
        """Pick a helper word for deterministic sentence construction."""
        words = self.get_unused_words_by_pos(pos)
        if words:
            return random.choice(words)
        all_pos_words = self.classifier.get_words_by_pos(pos)
        if all_pos_words:
            return random.choice(all_pos_words)
        return None

    def _construct_word_sentence(self, word: str, info: dict) -> Optional[str]:
        """Construct a simple but varied sentence based on word's POS."""
        pos = info.get('pos')

        if pos == 'VERB':
            # Use verb with varied subjects and a full clause to hit min length
            subjects = ["alguien", "la gente", "muchos", "algunos", "un grupo"]
            frames = [
                f"{random.choice(subjects)} suele {word} cada día",
                f"{random.choice(subjects)} busca {word} con calma",
                f"{random.choice(subjects)} decide {word} para avanzar",
            ]
            return random.choice(frames)
        elif pos == 'NOUN':
            # Use noun with varied verbs and structures
            gender = info.get('gender', 'Masc')
            number = info.get('number', 'Sing')
            if number == 'Plur':
                article = 'los' if gender == 'Masc' else 'las'
                verbs = [
                    "aportan valor a todos",
                    "crean impacto en la vida",
                    "mejoran la vida diaria",
                    "inspiran a la comunidad",
                    "protegen sueños cada día",
                ]
            else:
                article = 'el' if gender == 'Masc' else 'la'
                verbs = [
                    "aporta valor a todos",
                    "crea impacto en la vida",
                    "mejora la vida diaria",
                    "inspira a la comunidad",
                    "protege sueños cada día",
                ]
            return f"{article} {word} {random.choice(verbs)}"
        elif pos == 'ADJ':
            # Use adjective with varied copulas
            structures = [
                f"es claramente {word} para todos hoy",
                f"se siente muy {word} en verdad",
                f"resulta siempre {word} para muchos",
            ]
            return random.choice(structures)
        elif pos == 'ADV':
            # Use adverb with varied verbs
            verbs = [
                "orienta nuestras acciones cada día",
                "marca el ritmo de la jornada",
                "guía decisiones en la vida",
            ]
            return f"{word} {random.choice(verbs)}"
        elif pos in ('PROPN', 'PRON'):
            # Proper noun or pronoun with varied verbs
            verbs = [
                "ofrece apoyo a la comunidad",
                "guía proyectos en la región",
                "crea oportunidades para muchos",
                "inspira confianza cada día",
            ]
            return f"{word} {random.choice(verbs)}"

        return None

    def _default_word_sentence(self, word: str, info: dict) -> str:
        """Fallback sentence when construct fails."""
        # Use a non-trivial verb phrase to avoid "existe" and meet min length
        return f"{word} aporta valor en la vida diaria"

    def _build_required_word_fallback(self, required_words: List[str], template: Template) -> Optional[tuple[str, List[str]]]:
        """Construct a deterministic sentence when the LLM cannot include the required word."""
        if not required_words:
            return None
        primary = required_words[0]
        info = self.classifier.get_word_info(primary) or {}
        sentence = self._construct_word_sentence(primary, info)
        if not sentence:
            sentence = self._default_word_sentence(primary, info)
        formatted = self._format_sentence(sentence, template)
        used_words = [primary]
        # Screen fallback for length/trivial issues; if it fails, return None to trigger a different path
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", formatted)
        word_count = len(tokens)
        lower_sentence = formatted.lower()
        if word_count < self.max_words_min or word_count > self.max_words_allowed:
            self._log_debug(f"Constructed fallback rejected for length ({word_count} words): {formatted}")
            return None
        if re.search(r'\b(existe|existen|est[áa] presente|est[áa]n presentes|se encuentran aqu[íi]|es importante|son importantes|son importante)\b', lower_sentence):
            self._log_debug(f"Constructed fallback rejected as trivial: {formatted}")
            return None
        return formatted, used_words

    def _generate_with_llm(self, template: Template, mark_used: bool = True, required_words: Optional[List[str]] = None) -> Optional[tuple[str, List[str]]]:
        """
        Ask LLM to generate a sentence when template-based generation fails.

        Args:
            template: Template to use as guide
            mark_used: Whether to mark words as used
            required_words: Specific words that must appear in the sentence

        Returns:
            Tuple of (sentence, list of words used) or None if failed
        """
        if not self.generation_provider:
            return None

        required_words = required_words or []

        # Get unused words by POS needed for template
        available_words = {}
        for pos in set(template.slots):
            words = self.get_unused_words_by_pos(pos)
            if not words:
                words = self.classifier.get_words_by_pos(pos)
            if words:
                # Use configured word pool size for LLM generation
                if len(words) > self.LLM_WORD_POOL_SIZE:
                    selected_words = random.sample(words, self.LLM_WORD_POOL_SIZE)
                else:
                    selected_words = list(words)
                for req in required_words:
                    info = self.classifier.get_word_info(req)
                    if info.get('pos') == pos and req not in selected_words:
                        selected_words.append(req)
                available_words[pos] = selected_words

        if not available_words:
            self._log_debug(f"No available words for template {getattr(template, 'name', template)}; skipping LLM generation")
            return None

        # Select seed topic based on available words
        seed_topic = self._select_seed_topic(available_words)

        is_rescue_template = hasattr(template, "name") and template.name.startswith("RESCUE_")

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

        prompt = f"""⚠️ CRITICAL: NO MORE THAN {self.max_words} WORDS. Count carefully.

{base_instruction}

"""
        for pos, words in available_words.items():
            prompt += f"{pos}: {', '.join(words)}\n"

        prompt += f"""
Requirements:
- Maximum {self.max_words} words total
- MUST be a COMPLETE sentence with both subject and verb (reject fragments like "Acondicionado informe" or "Análisis verlo")
- Use as many words from the list as fit naturally (skip 1-2 if needed)
- Sentence MUST make logical sense and sound natural to a native Spanish speaker
- MUST be grammatically correct: proper gender agreement, correct verb conjugation, proper word order
- AVOID nonsensical combinations (e.g., "Agua cerrará", "Almirante Consiste")
- DO NOT start with "Espero que" - use varied sentence structures"""

        if is_rescue_template:
            prompt += """
- Do NOT output trivial sentences like "X existe/está presente/están aquí" or "X es/está <adjective>".
- Include a meaningful action or context around the required word, with a clear subject + verb + an additional phrase (reason, location, time, or purpose).
- Prefer using an article before nouns when natural ("el/la/los/las <word>")."""

        if required_words:
            prompt += f"\n- CRITICAL: Sentence MUST include each of these exact words: {', '.join(required_words)}"

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

        # Try multiple times to get a valid sentence before falling back
        for attempt in range(self.LLM_MAX_ATTEMPTS):
            try:
                # Ask LLM to generate
                if hasattr(self.llm_validator, '_check_ollama') and (self.generation_provider in (None, 'ollama')):
                    payload = {
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5 + (attempt * 0.15),  # Start lower for better instruction following
                            # Keep concise but give enough room to avoid empty replies
                            "num_predict": max(24, int(self.max_words * 2.5))
                        }
                    }
                    response = requests.post(self.llm_validator.url, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    sentence = result.get('response', '').strip()
                else:
                    # Keep concise but give enough room to avoid empty replies
                    max_tokens = max(24, int(self.max_words * 2.4))
                    if self.generation_provider == 'openai' or (self.llm_validator and self.llm_validator.provider == 'openai'):
                        sentence = self.llm_validator.generate_openai(
                            prompt,
                            temperature=0.5 + (attempt * 0.15),
                            max_tokens=max_tokens,
                            model_override=self.generation_model
                        )
                    elif self.generation_provider == 'anthropic' or (self.llm_validator and self.llm_validator.provider == 'anthropic'):
                        sentence = self.llm_validator.generate_anthropic(
                            prompt,
                            temperature=0.5 + (attempt * 0.15),
                            max_tokens=max_tokens,
                            model_override=self.generation_model
                        )
                    else:
                        return None
            except Exception as e:
                # On generation error, record and retry
                print(f"\n⚠️  LLM generation error (attempt {attempt + 1}/{self.LLM_MAX_ATTEMPTS}): {e}")
                self._log_debug(f"LLM generation error (attempt {attempt + 1}): {e}")
                self.rejection_counts['validator'] += 1
                self._write_stats_snapshot()
                continue

            # Clean up sentence - extract only the first line (ignore LLM chatter)
            raw_sentence = sentence
            sentence = sentence.strip().strip('"').strip("'")
            # Take only the first line if LLM added extra text
            if '\n' in sentence:
                sentence = sentence.split('\n')[0].strip()
            # Remove common LLM artifacts
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
                self._log_debug(f"LLM returned empty after cleanup (attempt {attempt + 1}); raw='{raw_sentence}'")
                continue  # Try again

            self._log_debug(f"LLM raw sentence (attempt {attempt + 1}): {sentence}")

            # Format sentence
            formatted_sentence = self._format_sentence(sentence, template)
            # Optional post-fix via LLM to clean grammar/capitalization
            fixed_sentence = self._fix_sentence_with_llm(formatted_sentence)
            if fixed_sentence:
                formatted_sentence = self._format_sentence(fixed_sentence, template)
            # Track for logging after any fixes
            self._current_candidate_sentence = formatted_sentence

            # Count words once for downstream checks
            tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", formatted_sentence)
            word_count = len(tokens)

            # Enforce word count bounds first; try shortening if too long (up to allowed slack)
            if word_count < self.max_words_min:
                self._record_rejection('too_short')
                self._log_debug(f"Rejected too_short ({word_count} words): {formatted_sentence}")
                continue
            if word_count > self.max_words_allowed:
                shortened = self._shorten_sentence_with_llm(formatted_sentence)
                if shortened:
                    formatted_sentence = self._format_sentence(shortened, template)
                    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", formatted_sentence)
                    word_count = len(tokens)
                if word_count > self.max_words_allowed or word_count < self.max_words_min:
                    self._record_rejection('too_long' if word_count > self.max_words_allowed else 'too_short')
                    self._log_debug(f"Rejected length after shorten ({word_count} words): {formatted_sentence}")
                    continue
            # Note: If between target and allowed, we still accept but will log below in _commit_sentence

            # Reject trivial copula/filler outputs (use final sentence text)
            lower_sentence = formatted_sentence.lower()
            if re.search(r'\b(existe|existen|est[áa] presente|est[áa]n presentes|se encuentran aqu[íi]|es importante|son importantes|son importante)\b', lower_sentence):
                self._record_rejection('trivial')
                self._log_debug(f"Rejected trivial: {formatted_sentence}")
                continue

            if self.spellcheck_enabled:
                spell_start = time.perf_counter()
                has_spelling_errors, _ = self._has_spelling_errors(formatted_sentence)
                self.spellcheck_duration += time.perf_counter() - spell_start
                if has_spelling_errors:
                    self._record_rejection('spelling')
                    continue

            similarity_features = None
            start_raw, start_normalized = self._extract_sentence_start(formatted_sentence)
            start_bigram = self._compute_start_bigram(formatted_sentence)

            if self._is_recent_start(start_raw, start_normalized, start_bigram):
                self._record_rejection('start_repeat')
                continue

            # Validate grammar using LLM validator
            if not self._passes_semantic_heuristics(formatted_sentence, template):
                self._current_candidate_sentence = None
                continue

            similarity_features = self._build_similarity_features(
                formatted_sentence,
                start_data=(start_raw, start_normalized, start_bigram)
            )
            repetitive, similarity_detail = self._is_repetitive_via_similarity(similarity_features)
            self._update_similarity_stats(similarity_detail, repetitive)
            self._record_similarity_score(similarity_detail.get('combined'))
            if repetitive:
                self._record_rejection('similarity_combined')
                continue

            llm_ok = True
            if not self.skip_llm_validation and self.llm_validator:
                llm_ok = self.llm_validator.is_coherent(formatted_sentence)
                self._log_debug(f"Validator {'accepted' if llm_ok else 'rejected'}: {formatted_sentence}")

            if llm_ok:
                if similarity_features is None:
                    similarity_features = self._build_similarity_features(
                        formatted_sentence,
                        start_data=(start_raw, start_normalized, start_bigram)
                    )
                # Extract words used from sentence
                words_used = []
                # Always count required words as used (even if inflected/adjusted)
                if required_words:
                    for required in required_words:
                        if required not in words_used:
                            words_used.append(required)
                sentence_lower = sentence.lower()

                for pos, words in available_words.items():
                    for word in words:
                        if word.lower() in sentence_lower and word not in words_used:
                            words_used.append(word)

                commit_words = words_used if mark_used else None
                self._commit_sentence(
                    formatted_sentence,
                    template,
                    commit_words,
                    similarity_features=similarity_features,
                    start_data=(start_raw, start_normalized, start_bigram)
                )
                self._current_candidate_sentence = None
                return (formatted_sentence, words_used)
            # If validation failed, try again with next attempt
            self.rejection_counts['validator'] += 1
            template_id = getattr(template, 'identifier', template.name if hasattr(template, 'name') else str(template))
            self.template_fail_counts[template_id] += 1
            self._write_stats_snapshot()
            # Log validator rejection
            if self._rejection_log is not None and self._current_candidate_sentence:
                self._rejection_log.append({
                    'reason': 'validator',
                    'sentence': self._current_candidate_sentence,
                    'timestamp': time.time()
                })
                self._log_debug(f"Validator rejected: {self._current_candidate_sentence}")
                self._flush_rejection_log()
            self._current_candidate_sentence = None

        # All attempts failed validation; try deterministic fallback
        self._log_debug(f"LLM attempts exhausted for template {getattr(template, 'name', template)}; using deterministic fallback")
        fallback = self._build_required_word_fallback(required_words, template)
        if fallback:
            formatted_sentence, fallback_words = fallback
            commit_words = fallback_words if mark_used else None
            self._commit_sentence(formatted_sentence, template, commit_words)
            return (formatted_sentence, fallback_words)
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
            if ' pero ' in sentence_lower and ',' in sentence_lower:
                if not self._conditional_has_resolution(sentence_lower, require_resolution=True):
                    self._record_rejection('conditional_pero')
                    return False

        if template.mood == 'Imp':
            if not self._is_valid_imperative(sentence_lower):
                self._record_rejection('imperative')
                return False

        if any(color in sentence_lower for color in self.COLOR_WORDS):
            if not any(context in sentence_lower for context in self.COLOR_CONTEXT_WORDS):
                self._record_rejection('color_context')
                return False
            if any(abstract in sentence_lower for abstract in self.ABSTRACT_COLOR_WORDS):
                self._record_rejection('color_abstract')
                return False

        if self._has_repetitive_phrase(sentence_lower):
            self._record_rejection('repetition')
            return False

        return True

    def _has_spelling_errors(self, sentence: str) -> tuple[bool, List[str]]:
        """Return True if the sentence contains tokens not present in the dictionary."""
        if not (self.spell_checker and self.spellcheck_enabled):
            return False, []
        tokens = self.WORD_PATTERN.findall(sentence)
        if not tokens:
            return False, []
        normalized = []
        for token in tokens:
            cleaned = token.lower()
            if not cleaned:
                continue
            if cleaned in self._spellchecker_whitelist:
                continue
            normalized.append(cleaned)
        if not normalized:
            return False, []
        unknown = self.spell_checker.unknown(normalized)
        if not unknown:
            return False, []
        if self.spellcheck_enabled:
            for word in unknown:
                self.spellcheck_unknown_counts[word] += 1
        return True, sorted(unknown)

    def _has_repetitive_phrase(self, sentence_lower: str) -> bool:
        """Detect overuse of repeated words or phrases (generic heuristic)."""
        tokens = [t for t in sentence_lower.split() if t]
        if len(tokens) < 2:
            return False

        # Flag if a non-stopword (length > 3) appears 3+ times
        counter = Counter(tokens)
        for word, count in counter.items():
            if count >= 3 and len(word) > 3 and word not in self.REPETITION_STOPWORDS:
                return True

        # Flag repeated n-grams (2-4 words) that contain at least one non-stopword
        for n in range(2, 5):
            seen = {}
            for idx in range(len(tokens) - n + 1):
                phrase = tuple(tokens[idx : idx + n])
                if all(word in self.REPETITION_STOPWORDS for word in phrase):
                    continue
                if phrase in seen:
                    return True
                seen[phrase] = True
        return False

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
        # Log rejected sentence if available
        if self._rejection_log is not None and hasattr(self, '_current_candidate_sentence'):
            entry = {
                'reason': reason,
                'sentence': self._current_candidate_sentence,
                'timestamp': time.time()
            }
            self._rejection_log.append(entry)
            self._flush_rejection_log()
        self._write_stats_snapshot()

    def _record_duration_bucket(self, duration: float):
        """Bucket sentence generation durations for distribution stats."""
        if duration < 4:
            bucket = "<4s"
        elif duration < 8:
            bucket = "4-8s"
        elif duration < 12:
            bucket = "8-12s"
        elif duration < 16:
            bucket = "12-16s"
        elif duration < 20:
            bucket = "16-20s"
        else:
            bucket = ">=20s"
        self.duration_histogram[bucket] += 1

    def _commit_sentence(
        self,
        formatted_sentence: str,
        template: Template,
        words_used: Optional[List[str]],
        similarity_features: Optional[dict] = None,
        start_data: Optional[Tuple[Optional[str], Optional[str], str]] = None
    ):
        """Register a fully accepted sentence (append starts, similarity, mark words)."""
        word_count = len(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", formatted_sentence))
        over_target = ""
        if word_count > self.max_words:
            over_target = f" (over target {self.max_words}, allowed {self.max_words_allowed})"
        self._log_debug(f"Accepted ({word_count} words){over_target}: {formatted_sentence}")
        if start_data:
            start_raw, start_normalized, start_bigram = start_data
        else:
            start_raw, start_normalized = self._extract_sentence_start(formatted_sentence)
            start_bigram = self._compute_start_bigram(formatted_sentence)
        self._append_sentence_start(start_raw, start_normalized, start_bigram)
        if similarity_features is None:
            similarity_features = self._build_similarity_features(
                formatted_sentence,
                start_data=(start_raw, start_normalized, start_bigram)
            )
        self._register_sentence_similarity(similarity_features)
        if words_used:
            for word in words_used:
                self.mark_word_used(word)
        self.llm_generated_count += 1
        pattern = self._template_pattern_signature(template)
        if pattern:
            self._recent_template_patterns.append(pattern)
            if len(self._recent_template_patterns) > 40:
                self._recent_template_patterns.pop(0)
        self._write_stats_snapshot()

    def _consume_word_directly(self, word: str) -> Optional[tuple[str, List[str]]]:
        """Use a deterministic fallback sentence for arbitrary words."""
        if not word:
            return None
        fallback = self._build_required_word_fallback([word], self._fallback_template)
        if fallback:
            formatted_sentence, words_used = fallback
        else:
            sentence = self._default_word_sentence(word, {})
            formatted_sentence = self._format_sentence(sentence, self._fallback_template)
            words_used = [word]
        # Ensure fallback also respects length/trivial guards
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", formatted_sentence)
        word_count = len(tokens)
        if word_count < self.max_words_min or word_count > self.max_words_allowed:
            self._log_debug(f"Fallback rejected for length ({word_count} words): {formatted_sentence}")
            return None
        lower_sentence = formatted_sentence.lower()
        if re.search(r'\b(existe|existen|est[áa] presente|est[áa]n presentes|se encuentran aqu[íi]|es importante|son importantes|son importante)\b', lower_sentence):
            self._log_debug(f"Fallback rejected as trivial: {formatted_sentence}")
            return None
        self._commit_sentence(formatted_sentence, self._fallback_template, words_used)
        return formatted_sentence, words_used

    def _build_rescue_template(self, word: str) -> Optional[Template]:
        """
        Build a minimal template to give the LLM one last chance before fallback.
        Keeps structure simple (single slot) to reduce rejection risk.
        """
        info = self.classifier.get_word_info(word) if self.classifier else None
        pos = info.get('pos') if info else None
        if not pos:
            return None
        return Template(
            name=f"RESCUE_{pos}",
            slots=[pos],
            glue_words=[],
            pattern="slot0",
            mood=None
        )

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
            print(f"\nGenerating sentences (min {self.max_words_min}, max {self.max_words} words per sentence)...")
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
        batch_start_time = time.time()
        last_batch_count = 0
        self.last_run_aborted = False

        sentence_timer = None
        while self.unused_words:
            if self.max_sentences and len(sentences) >= self.max_sentences:
                if verbose:
                    print(f"\nReached max sentences limit ({self.max_sentences}). Stopping.")
                break
            iteration += 1
            if sentence_timer is None:
                sentence_timer = time.time()

            # Show progress immediately
            if verbose and iteration == 1:
                sys.stdout.write(f'\r  Generating sentence 1... 0.0s elapsed')
                sys.stdout.flush()

            # Select template - prefer templates that cover the most overdue POS
            target_pos = self._select_priority_pos()
            priority_word = None
            template = self._select_best_template(templates, target_pos, priority_word)

            required_words = None
            if template is not None and target_pos and target_pos in template.slots:
                priority_word = self._select_priority_word(target_pos)
                if priority_word:
                    required_words = [priority_word]

            # Generate sentence with LLM (required)
            if not self.generation_provider:
                if verbose:
                    print("\nError: Generation provider is required for sentence generation")
                break

            # Show progress before LLM call
            if verbose:
                elapsed = time.time() - batch_start_time
                # Clear line first, then write new message
                sys.stdout.write('\r' + ' ' * 140 + '\r')
                sys.stdout.write(f'  Generating sentence {len(sentences) + 1}... {elapsed:.1f}s elapsed (calling LLM...) | {self._format_rejection_progress()}')
                sys.stdout.flush()

            result = None
            used_template = template
            if template:
                result = self._generate_with_llm(template, required_words=required_words)
            if not result:
                rescue_word = priority_word or self._select_any_unused_word()
                rescue_template = self._build_rescue_template(rescue_word) if rescue_word else None
                if rescue_template and rescue_word:
                    self._log_debug(f"Rescue template triggered for word '{rescue_word}'")
                    result = self._generate_with_llm(rescue_template, required_words=[rescue_word])
                    if result:
                        used_template = rescue_template
            if not result:
                # If a targeted word is struggling, postpone it briefly instead of consuming immediately
                if priority_word:
                    self._stalled_words[priority_word] += 1
                    if self._stalled_words[priority_word] < 3:
                        # Try other words/templates first; avoid marking this one used yet
                        sentence_timer = None
                        continue
                fallback_word = priority_word or self._select_any_unused_word()
                if fallback_word:
                    self._log_debug(f"Falling back deterministically for word '{fallback_word}'")
                    result = self._consume_word_directly(fallback_word)
                    used_template = self._fallback_template
            if result:
                sentence, words_used = result
                self.last_sentence_duration = time.time() - sentence_timer
                sentence_timer = None
                if self.first_sentence_duration is None and self.successful_sentences == 0:
                    self.first_sentence_duration = self.last_sentence_duration
                self._record_duration_bucket(self.last_sentence_duration)
                self.total_sentence_duration += self.last_sentence_duration
                self.successful_sentences += 1
                sentences.append(sentence)
                template_for_count = used_template or template or self._fallback_template
                template_id = getattr(template_for_count, 'identifier', template_for_count.name if hasattr(template_for_count, 'name') else str(template_for_count))
                self.template_success_counts[template_id] += 1

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
                if verbose:
                    print(f"\nWarning: Unable to generate sentences for remaining {len(self.unused_words)} words.")
                break

        if verbose:
            print(f"\nGeneration complete!")
            stats_summary = self.get_statistics()
            total_rejections = self.rejection_counts.get('heuristic', 0) + self.rejection_counts.get('validator', 0)
            print(self._format_two_columns("Total sentences", str(len(sentences)), "Total rejections", str(total_rejections)))
            print(self._format_two_columns("Words covered", str(len(self.used_words)), "Heuristic rejects", str(self.rejection_counts.get('heuristic', 0))))
            print(self._format_two_columns("Words remaining", str(len(self.unused_words)), "Validator rejects", str(self.rejection_counts.get('validator', 0))))
            print(self._format_two_columns("Coverage", f"{stats_summary['coverage_percent']:.2f}%", "Last sentence time", f"{stats_summary['last_sentence_duration']:.2f}s"))
            print(self._format_two_columns("Avg sentence time", f"{stats_summary['avg_sentence_duration']:.2f}s", "Max sentences", str(self.max_sentences) if self.max_sentences else "All words"))
            spell_total_ms = stats_summary.get('spellcheck_duration', 0.0) * 1000.0
            spell_avg_ms = (spell_total_ms / len(sentences)) if sentences else 0.0
            if self.spellcheck_enabled:
                print(self._format_two_columns("Spell check total", f"{spell_total_ms:.2f} ms", "Spell check avg", f"{spell_avg_ms:.2f} ms"))
            else:
                print(self._format_two_columns("Spell check", "disabled"))
            if self.spellcheck_enabled and stats_summary.get('spellcheck_unknowns'):
                print("\nTop spell-check rejects:")
                for word, count in stats_summary['spellcheck_unknowns'][:5]:
                    print(f"  {word:<25}{count:>5}")
            if self.template_fail_counts:
                print("\nTop template failures:")
                sorted_templates = sorted(self.template_fail_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for template_id, count in sorted_templates:
                    print(f"  {template_id:<40}{count:>6}")
            if self.template_success_counts:
                print("\nTop template selections:")
                sorted_success = sorted(self.template_success_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for template_id, count in sorted_success:
                    print(f"  {template_id:<40}{count:>6}")
            histogram = stats_summary.get('duration_histogram', {})
            if histogram:
                print("\nSentence duration histogram:")
                for bucket in self.DURATION_BUCKETS:
                    if bucket in histogram:
                        print(f"  {bucket:<6}{histogram[bucket]:>6}")

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
        self._write_stats_snapshot()
        return sentences

    def _flush_rejection_log(self):
        """Persist rejection log entries immediately for real-time analysis."""
        if not self._rejection_log or not self._rejection_log_path:
            return
        try:
            with open(self._rejection_log_path, 'a', encoding='utf-8') as f:
                while self._rejection_log:
                    entry = self._rejection_log.pop(0)
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get('timestamp', time.time())))
                    f.write(f"[{ts}] {entry.get('reason','unknown')}: {entry.get('sentence','')}\n")
        except Exception:
            pass

    def _select_priority_pos(self) -> Optional[str]:
        """Return the POS tag with the largest number of unused words."""
        if not self._unused_pos_counts:
            return None
        prioritized = sorted(self._unused_pos_counts.items(), key=lambda x: x[1], reverse=True)
        for pos, count in prioritized:
            if count > 0 and (not self._template_pos_set or pos in self._template_pos_set):
                return pos
        return None

    def _select_priority_word(self, pos: str) -> Optional[str]:
        """Pick the least-used unused word of a given POS for targeted coverage."""
        candidates = self.get_unused_words_by_pos(pos)
        if not candidates:
            return None
        non_stalled = [w for w in candidates if self._stalled_words.get(w, 0) == 0]
        pool = non_stalled if non_stalled else candidates
        return min(pool, key=lambda w: (self.word_usage_count.get(w, 0), self._stalled_words.get(w, 0), w))

    def _select_any_unused_word(self) -> Optional[str]:
        """Return any remaining unused word regardless of POS."""
        if not self.unused_words:
            return None
        return next(iter(self.unused_words))

    def _analyze_word_traits(self, word: Optional[str]) -> dict:
        """Return cached traits (mood, lower, question-flag) for a word."""
        traits = {}
        if not word or not self.classifier:
            return traits
        info = self.classifier.get_word_info(word) or {}
        mood = info.get('mood')
        if mood:
            traits['mood'] = mood
        lower = word.lower()
        traits['lower'] = lower
        traits['is_question'] = lower in self.QUESTION_WORDS
        return traits

    def _template_word_adjustment(
        self,
        template: Template,
        template_mood: str,
        is_question_template: bool,
        word_traits: dict
    ) -> Optional[float]:
        """Return score adjustment for compatibility or None if incompatible."""
        if not word_traits:
            return 0.0
        lower = word_traits.get('lower')
        mood = word_traits.get('mood')
        is_question_word = word_traits.get('is_question', False)
        template_mood = template_mood or (template.mood or 'Ind')

        if template.mood == 'Imp':
            if lower in self.imperative_forms or (mood == 'Imp'):
                return 80.0
            return None
        if template_mood == 'Question':
            if is_question_word:
                return 80.0
            return -80.0
        if is_question_word and not is_question_template:
            return -60.0
        if template.mood == 'Sub':
            if mood == 'Sub':
                return 40.0
            if mood and mood != 'Sub':
                return -40.0
        if mood == 'Imp' and template.mood not in (None, 'Imp'):
            return -20.0
        return 0.0

    def _select_best_template(
        self,
        templates: List[Template],
        target_pos: Optional[str] = None,
        target_word: Optional[str] = None
    ) -> Optional[Template]:
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

        candidate_templates = templates
        if target_pos:
            targeted = [template for template in templates if target_pos in template.slots]
            if targeted:
                candidate_templates = targeted
        word_traits = self._analyze_word_traits(target_word)

        # Score each template based on how many unused words it can use
        scored_templates = []
        total_words = len(self.used_words) + len(self.unused_words)
        coverage = (len(self.used_words) / total_words) if total_words else 0.0
        early_run = len(self.sentences) < 50 or coverage < 0.05
        moderate_run = coverage < 0.15

        for template in candidate_templates:
            score = 0
            can_fill = True

            for pos in template.slots:
                total_for_pos = len(self.classifier.get_words_by_pos(pos))
                unused_count = self._unused_pos_counts.get(pos, 0)

                if unused_count > 0:
                    coverage_ratio = unused_count / max(total_for_pos, 1)
                    score += 10 + coverage_ratio * 60 + min(unused_count, 200) * 0.2
                    if target_pos and pos == target_pos:
                        score += 150
                elif not total_for_pos:
                    can_fill = False
                    break
                else:
                    score -= 15

            if can_fill:
                # Add variety bonus for underused moods
                template_mood = template.mood or 'Ind'

                # Check if it's a question
                is_question = any(q in template.glue_words for q in ['¿qué', '¿dónde', '¿cuándo', '¿cómo', '¿quién', '¿por qué'])
                if is_question:
                    template_mood = 'Question'

                compatibility_adjustment = self._template_word_adjustment(
                    template,
                    template_mood,
                    is_question,
                    word_traits
                )
                if compatibility_adjustment is None:
                    continue
                score += compatibility_adjustment

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

                # Early in the run, down-rank the longest templates to reduce rejects
                if early_run:
                    slots_len = len(template.slots)
                    if slots_len >= 6:
                        score -= 120
                    elif slots_len >= 5:
                        score -= 80
                    elif slots_len >= 4:
                        score -= 40

                # Downrank high-failure conjunction templates (aunque/porque) until coverage improves
                glue_lower = [g.lower() for g in template.glue_words]
                if moderate_run:
                    if 'aunque' in glue_lower:
                        score -= 70
                    if 'porque' in glue_lower:
                        score -= 40

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

        effective_total = self.total_sentence_duration
        effective_count = self.successful_sentences
        if self.first_sentence_duration is not None:
            if effective_count > 1:
                effective_total = max(0.0, effective_total - self.first_sentence_duration)
                effective_count -= 1
            else:
                effective_total = 0.0
                effective_count = 0
        avg_sentence_duration = (effective_total / effective_count) if effective_count > 0 else 0.0

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
            'avg_sentence_duration': avg_sentence_duration,
            'duration_histogram': dict(self.duration_histogram),
            'spellcheck_duration': self.spellcheck_duration
        }

        # Add mood statistics if available
        if mood_stats:
            stats['mood_counts'] = mood_stats

        if self.spellcheck_unknown_counts:
            top_unknowns = sorted(self.spellcheck_unknown_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['spellcheck_unknowns'] = top_unknowns
        if self._similarity_guard_stats:
            stats['similarity_guard'] = dict(self._similarity_guard_stats)

        return stats
    def _fix_sentence_with_llm(self, sentence: str) -> Optional[str]:
        """
        Ask the LLM to correct grammar (gender/number, tense, articles) while preserving meaning.
        """
        if not self.llm_validator or not sentence or not self.generation_provider:
            return sentence
        prompt = f"""Corrige esta oración en español manteniendo su idea original.
- Arregla concordancia de género y número.
- Corrige tiempos verbales y artículos/preposiciones faltantes.
- No añadas ni quites información significativa.
- Devuelve solo la oración corregida, sin explicaciones ni formato.

Oración: {sentence}
Oración corregida:"""
        try:
            if self.generation_provider == 'openai' or (self.llm_validator and self.llm_validator.provider == 'openai'):
                return self.llm_validator.generate_openai(prompt, temperature=0.2, max_tokens=32, model_override=self.generation_model)
            if self.generation_provider == 'anthropic' or (self.llm_validator and self.llm_validator.provider == 'anthropic'):
                return self.llm_validator.generate_anthropic(prompt, temperature=0.2, max_tokens=64, model_override=self.generation_model)
        except Exception:
            return sentence
        return sentence
