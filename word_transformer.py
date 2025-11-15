"""
Word Transformer Module
Transforms Spanish words to match required grammatical features.
"""
from typing import Optional, Dict
import spacy


class WordTransformer:
    """Transforms Spanish words to match grammatical requirements."""
    
    def __init__(self, nlp):
        """
        Initialize the word transformer.
        
        Args:
            nlp: spaCy language model
        """
        self.nlp = nlp
        
        # Common irregular verb stems for present tense
        self.irregular_verbs = {
            'ser': {'3s': 'es', '3p': 'son'},
            'estar': {'3s': 'está', '3p': 'están'},
            'tener': {'3s': 'tiene', '3p': 'tienen'},
            'hacer': {'3s': 'hace', '3p': 'hacen'},
            'ir': {'3s': 'va', '3p': 'van'},
            'poder': {'3s': 'puede', '3p': 'pueden'},
            'decir': {'3s': 'dice', '3p': 'dicen'},
            'dar': {'3s': 'da', '3p': 'dan'},
            'saber': {'3s': 'sabe', '3p': 'saben'},
            'querer': {'3s': 'quiere', '3p': 'quieren'},
            'venir': {'3s': 'viene', '3p': 'vienen'},
            'ver': {'3s': 've', '3p': 'ven'},
            'poner': {'3s': 'pone', '3p': 'ponen'},
            'salir': {'3s': 'sale', '3p': 'salen'},
            'traer': {'3s': 'trae', '3p': 'traen'},
            'caer': {'3s': 'cae', '3p': 'caen'},
            'oír': {'3s': 'oye', '3p': 'oyen'},
        }
        
        # Common Spanish article patterns
        self.articles = {
            ('Masc', 'Sing', True): 'el',
            ('Fem', 'Sing', True): 'la',
            ('Masc', 'Plur', True): 'los',
            ('Fem', 'Plur', True): 'las',
            ('Masc', 'Sing', False): 'un',
            ('Fem', 'Sing', False): 'una',
            ('Masc', 'Plur', False): 'unos',
            ('Fem', 'Plur', False): 'unas',
        }
        
        # Common adjective endings by gender/number
        self.adj_endings = {
            ('Masc', 'Sing'): 'o',
            ('Fem', 'Sing'): 'a',
            ('Masc', 'Plur'): 'os',
            ('Fem', 'Plur'): 'as',
        }
    
    def transform_adjective(self, adjective: str, target_gender: str, target_number: str) -> str:
        """
        Transform an adjective to match target gender and number.
        
        Args:
            adjective: The adjective to transform
            target_gender: Target gender (Masc/Fem)
            target_number: Target number (Sing/Plur)
            
        Returns:
            Transformed adjective
        """
        # If adjective already ends in a consonant or 'e', it's likely invariable
        if adjective.endswith(('e', 'l', 'r', 'n', 'z', 's', 'd')):
            # Some adjectives are invariable, just handle plural
            if target_number == 'Plur' and not adjective.endswith('s'):
                return adjective + 's'
            return adjective
        
        # Remove current ending
        if adjective.endswith(('o', 'a', 'os', 'as')):
            if adjective.endswith(('os', 'as')):
                base = adjective[:-2]
            else:
                base = adjective[:-1]
        else:
            base = adjective
        
        # Add appropriate ending
        target_ending = self.adj_endings.get((target_gender, target_number), 'o')
        return base + target_ending
    
    def transform_noun_number(self, noun: str, target_number: str) -> str:
        """
        Transform a noun to match target number (singular/plural).
        
        Args:
            noun: The noun to transform
            target_number: Target number (Sing/Plur)
            
        Returns:
            Transformed noun
        """
        if target_number == 'Plur':
            # Make plural
            if noun.endswith('s'):
                return noun  # Already plural or invariable
            elif noun.endswith('z'):
                return noun[:-1] + 'ces'
            elif noun.endswith(('a', 'e', 'i', 'o', 'u')):
                return noun + 's'
            else:
                return noun + 'es'
        else:
            # Make singular (simple heuristic)
            if noun.endswith('ces'):
                return noun[:-3] + 'z'
            elif noun.endswith('es'):
                return noun[:-2]
            elif noun.endswith('s') and len(noun) > 2:
                return noun[:-1]
            return noun
    
    def conjugate_verb_for_subject(self, verb: str, subject_number: str, 
                                   target_mood: str = 'Ind') -> Optional[str]:
        """
        Conjugate a verb for a subject using simple rule-based conjugation.
        
        Args:
            verb: The verb (any form)
            subject_number: Subject number (Sing/Plur)
            target_mood: Target mood (Ind/Imp/Sub)
            
        Returns:
            Conjugated verb or None if can't conjugate
        """
        try:
            # Get the lemma (infinitive form)
            doc = self.nlp(verb)
            if len(doc) == 0:
                return None
            
            lemma = doc[0].lemma_
            person = '3s' if subject_number == 'Sing' else '3p'
            
            # Only support indicative mood for now (most common)
            if target_mood != 'Ind':
                return None
            
            # Check irregular verbs first
            if lemma in self.irregular_verbs:
                return self.irregular_verbs[lemma].get(person)
            
            # Regular verb conjugation rules for present indicative
            if lemma.endswith('ar'):
                stem = lemma[:-2]
                if person == '3s':
                    return stem + 'a'
                else:  # 3p
                    return stem + 'an'
            
            elif lemma.endswith('er'):
                stem = lemma[:-2]
                if person == '3s':
                    return stem + 'e'
                else:  # 3p
                    return stem + 'en'
            
            elif lemma.endswith('ir'):
                stem = lemma[:-2]
                if person == '3s':
                    return stem + 'e'
                else:  # 3p
                    return stem + 'en'
            
            return None
            
        except Exception:
            # If conjugation fails, return None
            return None
    
    def get_compatible_form(self, word: str, pos: str, 
                           target_gender: Optional[str] = None,
                           target_number: Optional[str] = None,
                           target_mood: Optional[str] = None) -> str:
        """
        Get a compatible form of a word matching target features.
        
        Args:
            word: The word to transform
            pos: Part of speech
            target_gender: Target gender (for adjectives)
            target_number: Target number (for nouns/adjectives/verbs)
            target_mood: Target mood (for verbs)
            
        Returns:
            Transformed word
        """
        if pos == 'ADJ' and target_gender and target_number:
            return self.transform_adjective(word, target_gender, target_number)
        
        elif pos in ['NOUN', 'PROPN'] and target_number:
            return self.transform_noun_number(word, target_number)
        
        elif pos == 'VERB' and target_number:
            result = self.conjugate_verb_for_subject(word, target_number, target_mood or 'Ind')
            return result if result else word
        
        return word
