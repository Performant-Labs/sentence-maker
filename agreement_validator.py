"""
Agreement Validator Module
Validates grammatical agreement in Spanish sentences.
"""
from typing import Dict, List, Optional, Tuple


class AgreementValidator:
    """Validates gender and number agreement in Spanish."""
    
    # Spanish articles with their gender/number
    ARTICLES = {
        'el': {'gender': 'Masc', 'number': 'Sing'},
        'la': {'gender': 'Fem', 'number': 'Sing'},
        'los': {'gender': 'Masc', 'number': 'Plur'},
        'las': {'gender': 'Fem', 'number': 'Plur'},
        'un': {'gender': 'Masc', 'number': 'Sing'},
        'una': {'gender': 'Fem', 'number': 'Sing'},
        'unos': {'gender': 'Masc', 'number': 'Plur'},
        'unas': {'gender': 'Fem', 'number': 'Plur'},
    }
    
    def __init__(self, word_classifier):
        """
        Initialize validator with word classifier.
        
        Args:
            word_classifier: WordClassifier instance with word info
        """
        self.classifier = word_classifier
    
    def get_word_features(self, word: str) -> Dict:
        """
        Get grammatical features for a word.
        
        Args:
            word: The word to look up
            
        Returns:
            Dictionary with gender, number, and POS
        """
        info = self.classifier.get_word_info(word)
        return {
            'gender': info.get('gender'),
            'number': info.get('number'),
            'pos': info.get('pos')
        }
    
    def get_article_for_noun(self, noun: str, definite: bool = True) -> Optional[str]:
        """
        Get the correct article for a noun based on its gender/number.
        
        Args:
            noun: The noun
            definite: If True, use definite article (el/la), else indefinite (un/una)
            
        Returns:
            Appropriate article or None if gender/number unknown
        """
        features = self.get_word_features(noun)
        gender = features.get('gender')
        number = features.get('number')
        
        if not gender or not number:
            # Default to masculine singular if unknown
            gender = 'Masc'
            number = 'Sing'
        
        # Select article based on gender and number
        if definite:
            if gender == 'Masc' and number == 'Sing':
                return 'el'
            elif gender == 'Fem' and number == 'Sing':
                return 'la'
            elif gender == 'Masc' and number == 'Plur':
                return 'los'
            elif gender == 'Fem' and number == 'Plur':
                return 'las'
        else:
            if gender == 'Masc' and number == 'Sing':
                return 'un'
            elif gender == 'Fem' and number == 'Sing':
                return 'una'
            elif gender == 'Masc' and number == 'Plur':
                return 'unos'
            elif gender == 'Fem' and number == 'Plur':
                return 'unas'
        
        return 'el'  # Default fallback
    
    def check_noun_adjective_agreement(self, noun: str, adjective: str) -> bool:
        """
        Check if noun and adjective agree in gender and number.
        
        Args:
            noun: The noun
            adjective: The adjective
            
        Returns:
            True if they agree or if features are unknown
        """
        noun_features = self.get_word_features(noun)
        adj_features = self.get_word_features(adjective)
        
        noun_gender = noun_features.get('gender')
        noun_number = noun_features.get('number')
        adj_gender = adj_features.get('gender')
        adj_number = adj_features.get('number')
        
        # If either word lacks gender/number info, allow it (can't validate)
        if not noun_gender or not adj_gender:
            return True
        
        # Check gender agreement
        if noun_gender != adj_gender:
            return False
        
        # Check number agreement if both have it
        if noun_number and adj_number and noun_number != adj_number:
            return False
        
        return True
    
    def check_article_noun_agreement(self, article: str, noun: str) -> bool:
        """
        Check if article and noun agree in gender and number.
        
        Args:
            article: The article (el, la, un, una, etc.)
            noun: The noun
            
        Returns:
            True if they agree
        """
        if article not in self.ARTICLES:
            return True  # Not a known article, allow it
        
        article_features = self.ARTICLES[article]
        noun_features = self.get_word_features(noun)
        
        noun_gender = noun_features.get('gender')
        noun_number = noun_features.get('number')
        
        # If noun lacks features, allow it
        if not noun_gender or not noun_number:
            return True
        
        # Check agreement
        return (article_features['gender'] == noun_gender and 
                article_features['number'] == noun_number)
    
    def validate_word_sequence(self, words: List[str], pos_tags: List[str]) -> bool:
        """
        Validate agreement in a sequence of words.
        
        Args:
            words: List of words in order
            pos_tags: Corresponding POS tags
            
        Returns:
            True if all agreements are valid
        """
        for i in range(len(words) - 1):
            current_word = words[i]
            current_pos = pos_tags[i]
            next_word = words[i + 1]
            next_pos = pos_tags[i + 1]
            
            # Check article-noun agreement
            if current_word.lower() in self.ARTICLES and next_pos in ['NOUN', 'PROPN']:
                if not self.check_article_noun_agreement(current_word.lower(), next_word):
                    return False
            
            # Check noun-adjective agreement (Spanish: noun before adjective usually)
            if current_pos in ['NOUN', 'PROPN'] and next_pos == 'ADJ':
                if not self.check_noun_adjective_agreement(current_word, next_word):
                    return False
            
            # Check adjective-noun agreement (when adjective comes first)
            if current_pos == 'ADJ' and next_pos in ['NOUN', 'PROPN']:
                if not self.check_noun_adjective_agreement(next_word, current_word):
                    return False
            
            # Check subject-verb agreement
            if current_pos in ['NOUN', 'PROPN'] and next_pos == 'VERB':
                if not self.check_subject_verb_agreement(current_word, next_word):
                    return False
        
        return True
    
    def find_compatible_adjective(self, noun: str, adjectives: List[str]) -> Optional[str]:
        """
        Find an adjective that agrees with the noun.
        
        Args:
            noun: The noun to match
            adjectives: List of candidate adjectives
            
        Returns:
            Compatible adjective or None
        """
        noun_features = self.get_word_features(noun)
        noun_gender = noun_features.get('gender')
        noun_number = noun_features.get('number')
        
        for adj in adjectives:
            if self.check_noun_adjective_agreement(noun, adj):
                return adj
        
        return None
    
    def find_compatible_noun(self, adjective: str, nouns: List[str]) -> Optional[str]:
        """
        Find a noun that agrees with the adjective.
        
        Args:
            adjective: The adjective to match
            nouns: List of candidate nouns
            
        Returns:
            Compatible noun or None
        """
        for noun in nouns:
            if self.check_noun_adjective_agreement(noun, adjective):
                return noun
        
        return None
    
    def check_subject_verb_agreement(self, subject: str, verb: str) -> bool:
        """
        Check if subject and verb agree in person and number.
        
        Args:
            subject: The subject (noun or pronoun)
            verb: The verb
            
        Returns:
            True if they agree or if features are unknown
        """
        subject_features = self.get_word_features(subject)
        verb_info = self.classifier.get_word_info(verb)
        
        # Get verb features
        verb_person = verb_info.get('person')
        verb_number = verb_info.get('number')
        verb_form = verb_info.get('verbform')
        
        # If verb is infinitive, gerund, or participle, it doesn't need to agree
        if verb_form in ['Inf', 'Ger', 'Part']:
            return False  # We want finite verbs for subject-verb agreement
        
        # If verb is not finite, skip agreement check
        if verb_form != 'Fin':
            return True
        
        # Get subject number
        subject_number = subject_features.get('number')
        
        # Nouns are 3rd person by default
        subject_person = '3'
        
        # If we don't have verb features, we can't validate
        if not verb_person or not verb_number:
            return True  # Allow if we can't check
        
        # Check person agreement (nouns are always 3rd person)
        if verb_person != subject_person:
            return False
        
        # Check number agreement
        if subject_number and verb_number and subject_number != verb_number:
            return False
        
        return True
    
    def find_compatible_verb(self, subject: str, verbs: List[str]) -> Optional[str]:
        """
        Find a verb that agrees with the subject in person and number.
        
        Args:
            subject: The subject noun
            verbs: List of candidate verbs
            
        Returns:
            Compatible verb or None
        """
        for verb in verbs:
            if self.check_subject_verb_agreement(subject, verb):
                return verb
        
        return None
    
    def get_verb_number(self, verb: str) -> Optional[str]:
        """
        Get the number (Sing/Plur) of a verb.
        
        Args:
            verb: The verb
            
        Returns:
            'Sing', 'Plur', or None
        """
        verb_info = self.classifier.get_word_info(verb)
        return verb_info.get('number')
    
    def is_finite_verb(self, verb: str) -> bool:
        """
        Check if a verb is in finite form (conjugated).
        
        Args:
            verb: The verb to check
            
        Returns:
            True if verb is finite (conjugated)
        """
        verb_info = self.classifier.get_word_info(verb)
        verb_form = verb_info.get('verbform')
        return verb_form == 'Fin'
    
    def check_verb_mood(self, verb: str, required_mood: str) -> bool:
        """
        Check if a verb is in the required mood.
        
        Args:
            verb: The verb to check
            required_mood: Required mood (Ind, Imp, Sub)
            
        Returns:
            True if verb matches required mood or if mood is unknown
        """
        verb_info = self.classifier.get_word_info(verb)
        verb_mood = verb_info.get('mood')
        
        # If we don't have mood info, allow it
        if not verb_mood:
            return True
        
        # Check if mood matches
        return verb_mood == required_mood
    
    def find_verb_with_mood(self, verbs: List[str], required_mood: str, subject: str = None) -> Optional[str]:
        """
        Find a verb with the required mood (and optionally matching subject).
        
        Args:
            verbs: List of candidate verbs
            required_mood: Required mood (Ind, Imp, Sub)
            subject: Optional subject for agreement checking
            
        Returns:
            Compatible verb or None
        """
        for verb in verbs:
            # Check mood
            if not self.check_verb_mood(verb, required_mood):
                continue
            
            # Check subject agreement if subject provided
            if subject and not self.check_subject_verb_agreement(subject, verb):
                continue
            
            # Check if verb is finite (for Ind and Sub moods)
            if required_mood in ['Ind', 'Sub']:
                if not self.is_finite_verb(verb):
                    continue
            
            return verb
        
        return None
    
    def get_verb_mood(self, verb: str) -> Optional[str]:
        """
        Get the mood of a verb.
        
        Args:
            verb: The verb
            
        Returns:
            'Ind', 'Imp', 'Sub', or None
        """
        verb_info = self.classifier.get_word_info(verb)
        return verb_info.get('mood')
