"""
Word Classifier Module
Classifies Spanish words by part of speech and grammatical features.
"""
import spacy
from collections import defaultdict
from typing import Dict, List, Set


class WordClassifier:
    """Classifies and categorizes Spanish words using spaCy."""
    
    def __init__(self):
        """Initialize the classifier with Spanish language model."""
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            print("Error: Spanish model not found.")
            print("Please run: python -m spacy download es_core_news_sm")
            raise
        
        self.categories = defaultdict(list)
        self.word_info = {}
        
    def classify_words(self, words: List[str]) -> Dict[str, List[str]]:
        """
        Classify a list of words by part of speech.
        
        Args:
            words: List of Spanish words to classify
            
        Returns:
            Dictionary mapping POS tags to lists of words
        """
        print(f"Classifying {len(words)} words...")
        
        # Process words in batches for efficiency
        batch_size = 1000
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            docs = list(self.nlp.pipe(batch))
            
            for word, doc in zip(batch, docs):
                if len(doc) > 0:
                    token = doc[0]
                    pos = token.pos_
                    
                    # Extract morphological features
                    morph = token.morph
                    gender = morph.get("Gender")
                    number = morph.get("Number")
                    person = morph.get("Person")
                    tense = morph.get("Tense")
                    mood = morph.get("Mood")
                    verbform = morph.get("VerbForm")
                    
                    # Store word info with grammatical features
                    self.word_info[word] = {
                        'pos': pos,
                        'lemma': token.lemma_,
                        'morph': str(token.morph),
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
