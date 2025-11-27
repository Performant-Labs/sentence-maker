"""
Word Classifier Module
Classifies Spanish words by part of speech and grammatical features.
"""
import hashlib
import json
import os
import spacy
from collections import defaultdict
from typing import Dict, List, Set


class WordClassifier:
    """Classifies and categorizes Spanish words using spaCy."""
    
    def __init__(self):
        """Initialize the classifier with Spanish language model."""
        try:
            self.nlp = spacy.load("es_core_news_lg")
        except OSError:
            print("Error: Spanish model not found.")
            print("Please run: python -m spacy download es_core_news_lg")
            raise
        
        self.categories = defaultdict(list)
        self.word_info = {}
        # Manual corrections for spaCy POS quirks on our list
        self.pos_overrides = {
            "acondicionado": {
                "pos": "ADJ",
                "gender": "Masc",
                "number": "Sing",
            },
        }
        # Cache path (shared across runs to avoid re-tagging)
        self.cache_path = os.path.join(os.path.dirname(__file__), "words", "word_info_cache.json")
        
    def classify_words(self, words: List[str]) -> Dict[str, List[str]]:
        """
        Classify a list of words by part of speech.
        
        Args:
            words: List of Spanish words to classify
            
        Returns:
            Dictionary mapping POS tags to lists of words
        """
        checksum = self._compute_checksum(words)
        if self._load_cache(checksum):
            return dict(self.categories)

        print(f"Classifying {len(words)} words...")
        
        # Process words in batches for efficiency
        batch_size = 1000
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            docs = list(self.nlp.pipe(batch))
            
            for word, doc in zip(batch, docs):
                if len(doc) == 0:
                    continue

                token = doc[0]

                # Re-run on lowercase form to improve POS for capitalized isolates
                lc_doc = list(self.nlp.pipe([word.lower()]))[0]
                lc_token = lc_doc[0] if len(lc_doc) else token

                pos = lc_token.pos_

                # Extract morphological features
                morph = lc_token.morph
                gender = morph.get("Gender")
                number = morph.get("Number")
                person = morph.get("Person")
                tense = morph.get("Tense")
                mood = morph.get("Mood")
                verbform = morph.get("VerbForm")

                # Heuristic: participles mis-tagged as NOUN -> treat as ADJ
                if pos == "NOUN":
                    if (verbform and verbform[0] == "Part") or word.lower().endswith(("ado", "ada", "ido", "ida")):
                        pos = "ADJ"

                # Apply manual overrides for known mis-tags
                override = self.pos_overrides.get(word.lower())
                if override:
                    pos = override.get("pos", pos)
                    gender = [override["gender"]] if override.get("gender") else gender
                    number = [override["number"]] if override.get("number") else number

                # Store word info with grammatical features
                self.word_info[word] = {
                    'pos': pos,
                    'lemma': token.lemma_,
                    'morph': str(lc_token.morph),
                    'is_alpha': token.is_alpha,
                    'gender': gender[0] if gender else None,  # Masc, Fem, or None
                    'number': number[0] if number else None,  # Sing, Plur, or None
                    'person': person[0] if person else None,  # 1, 2, 3, or None
                    'tense': tense[0] if tense else None,  # Pres, Past, Fut, etc.
                    'mood': mood[0] if mood else None,  # Ind, Sub, Imp, etc.
                    'verbform': verbform[0] if verbform else None,  # Fin, Inf, Ger, Part
                }

                # Categorize by POS
                self.categories[pos].append(word)
        
        print(f"Classification complete:")
        for pos, word_list in sorted(self.categories.items()):
            print(f"  {pos}: {len(word_list)} words")

        self._save_cache(checksum)
        
        return dict(self.categories)
    
    def get_words_by_pos(self, pos: str) -> List[str]:
        """Get all words of a specific part of speech."""
        return self.categories.get(pos, [])
    
    def get_word_info(self, word: str) -> Dict:
        """Get grammatical information about a word."""
        return self.word_info.get(word, {})
    
    def get_nouns(self) -> List[str]:
        """Get all nouns."""
        return self.categories.get('NOUN', []) + self.categories.get('PROPN', [])
    
    def get_verbs(self) -> List[str]:
        """Get all verbs."""
        return self.categories.get('VERB', [])
    
    def get_adjectives(self) -> List[str]:
        """Get all adjectives."""
        return self.categories.get('ADJ', [])
    
    def get_adverbs(self) -> List[str]:
        """Get all adverbs."""
        return self.categories.get('ADV', [])
    
    def get_determiners(self) -> List[str]:
        """Get all determiners."""
        return self.categories.get('DET', [])
    
    def get_prepositions(self) -> List[str]:
        """Get all prepositions (ADP in spaCy)."""
        return self.categories.get('ADP', [])
    
    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all word categories."""
        return dict(self.categories)

    def _compute_checksum(self, words: List[str]) -> str:
        """Compute a stable checksum for the word list (order-independent)."""
        normalized = "\n".join(sorted(words))
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _load_cache(self, checksum: str) -> bool:
        """Try to load cached word_info/categories if checksum and model match."""
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("checksum") != checksum:
                return False
            model_name = self.nlp.meta.get("name")
            model_version = self.nlp.meta.get("version")
            if data.get("model") != model_name or data.get("model_version") != model_version:
                return False
            self.word_info = data.get("word_info", {})
            self.categories = defaultdict(list, data.get("categories", {}))
            print(f"Loaded cached classification ({model_name} {model_version}); {len(self.word_info)} words.")
            return True
        except Exception:
            return False

    def _save_cache(self, checksum: str):
        """Persist word_info/categories to speed up future runs."""
        try:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            model_name = self.nlp.meta.get("name")
            model_version = self.nlp.meta.get("version")
            payload = {
                "checksum": checksum,
                "model": model_name,
                "model_version": model_version,
                "word_info": self.word_info,
                "categories": dict(self.categories),
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass
