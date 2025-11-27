"""
Sentence Templates Module
Defines grammatical templates for Spanish sentence generation.
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Template:
    """Represents a sentence template with POS slots."""
    name: str
    slots: List[str]  # POS tags for each slot
    glue_words: List[str]  # Fixed words (articles, prepositions, etc.)
    pattern: str  # Pattern showing how to combine slots and glue words
    mood: str = None  # Required mood for verbs (Ind, Imp, Sub, or None for any)
    
    def max_words(self) -> int:
        """Calculate maximum words this template can generate."""
        return len(self.slots) + len(self.glue_words)


class TemplateLibrary:
    """Library of Spanish sentence templates."""
    
    def __init__(self):
        """Initialize template library with common Spanish patterns."""
        self.templates = self._create_templates()
    
    def _create_templates(self) -> List[Template]:
        """Create a comprehensive set of Spanish sentence templates."""
        templates = [
            # Simple subject-verb patterns
            Template(
                name="NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1"
            ),
            Template(
                name="NOUN_VERB_NOUN",
                slots=["NOUN", "VERB", "NOUN"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 glue1 slot2"
            ),
            
            # With adjectives
            Template(
                name="ADJ_NOUN_VERB",
                slots=["ADJ", "NOUN", "VERB"],
                glue_words=["el", "la"],
                pattern="glue0 slot1 slot0 slot2"
            ),
            Template(
                name="NOUN_ADJ_VERB",
                slots=["NOUN", "ADJ", "VERB"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 slot2"
            ),
            Template(
                name="NOUN_VERB_ADJ",
                slots=["NOUN", "VERB", "ADJ"],
                glue_words=["el", "la", "es", "está"],
                pattern="glue0 slot0 slot1 glue1 slot2"
            ),
            
            # With prepositions
            Template(
                name="NOUN_VERB_PREP_NOUN",
                slots=["NOUN", "VERB", "ADP", "NOUN"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 slot2 glue1 slot3"
            ),
            Template(
                name="NOUN_PREP_NOUN_VERB",
                slots=["NOUN", "ADP", "NOUN", "VERB"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 glue1 slot2 slot3"
            ),
            Template(
                name="NOUN_CON_NOUN",
                slots=["NOUN", "NOUN"],
                glue_words=["el", "la", "con", "el", "la"],
                pattern="glue0 slot0 glue2 glue3 slot1"
            ),
            Template(
                name="NOUN_Y_NOUN",
                slots=["NOUN", "NOUN"],
                glue_words=["el", "la", "y", "el", "la"],
                pattern="glue0 slot0 glue2 glue3 slot1"
            ),
            
            # With adverbs
            Template(
                name="NOUN_VERB_ADV",
                slots=["NOUN", "VERB", "ADV"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 slot2"
            ),
            Template(
                name="ADV_NOUN_VERB",
                slots=["ADV", "NOUN", "VERB"],
                glue_words=["el", "la"],
                pattern="slot0 glue0 slot1 slot2"
            ),
            
            # Complex patterns
            Template(
                name="NOUN_VERB_NOUN_ADJ",
                slots=["NOUN", "VERB", "NOUN", "ADJ"],
                glue_words=["el", "la", "un", "una"],
                pattern="glue0 slot0 slot1 glue1 slot2 slot3"
            ),
            Template(
                name="ADJ_NOUN_VERB_PREP_NOUN",
                slots=["ADJ", "NOUN", "VERB", "ADP", "NOUN"],
                glue_words=["el", "la"],
                pattern="glue0 slot1 slot0 slot2 slot3 glue1 slot4"
            ),
            Template(
                name="NOUN_ADJ_VERB_ADV",
                slots=["NOUN", "ADJ", "VERB", "ADV"],
                glue_words=["el", "la"],
                pattern="glue0 slot0 slot1 slot2 slot3"
            ),
            
            # Shorter patterns for difficult words
            Template(
                name="NOUN_VERB_SHORT",
                slots=["NOUN", "VERB"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="ADJ_NOUN",
                slots=["ADJ", "NOUN"],
                glue_words=[],
                pattern="slot1 slot0"
            ),
            Template(
                name="NOUN_ADJ_SHORT",
                slots=["NOUN", "ADJ"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="PRON_VERB",
                slots=["PRON", "VERB"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="PROPN_VERB",
                slots=["PROPN", "VERB"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="PROPN_ES_ADJ",
                slots=["PROPN", "ADJ"],
                glue_words=["es"],
                pattern="slot0 glue0 slot1"
            ),
            Template(
                name="ADV_VERB",
                slots=["ADV", "VERB"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            Template(
                name="VERB_ADV_SHORT",
                slots=["VERB", "ADV"],
                glue_words=[],
                pattern="slot0 slot1"
            ),
            
            # Single word patterns (for very difficult words)
            Template(
                name="SINGLE_NOUN",
                slots=["NOUN"],
                glue_words=["el", "la"],
                pattern="glue0 slot0"
            ),
            Template(
                name="SINGLE_VERB",
                slots=["VERB"],
                glue_words=[],
                pattern="slot0"
            ),
            Template(
                name="SINGLE_ADJ",
                slots=["ADJ"],
                glue_words=["es", "muy"],
                pattern="glue0 glue1 slot0"
            ),
            Template(
                name="SINGLE_ADV",
                slots=["ADV"],
                glue_words=[],
                pattern="slot0"
            ),
            
            # Patterns with determiners
            Template(
                name="DET_NOUN_VERB",
                slots=["DET", "NOUN", "VERB"],
                glue_words=[],
                pattern="slot0 slot1 slot2"
            ),
            Template(
                name="DET_NOUN_VERB_NOUN",
                slots=["DET", "NOUN", "VERB", "NOUN"],
                glue_words=["el", "la"],
                pattern="slot0 slot1 slot2 glue0 slot3"
            ),
            Template(
                name="PROPN_VERB_NOUN",
                slots=["PROPN", "VERB", "NOUN"],
                glue_words=[],
                pattern="slot0 slot1 slot2"
            ),
            Template(
                name="PRON_VERB_NOUN",
                slots=["PRON", "VERB", "NOUN"],
                glue_words=[],
                pattern="slot0 slot1 slot2"
            ),
            
            # Question patterns (interrogatives)
            Template(
                name="QUE_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Qué", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="DONDE_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Dónde", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="CUANDO_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Cuándo", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="COMO_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Cómo", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="QUIEN_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Quién", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="POR_QUE_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¿Por qué", "el", "la", "?"],
                pattern="glue0 slot0 glue1 slot1 glue3"
            ),
            Template(
                name="QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["¿Qué", "?"],
                pattern="glue0 slot0 slot1 glue1"
            ),
            Template(
                name="DONDE_ESTA_NOUN",
                slots=["NOUN"],
                glue_words=["¿Dónde", "está", "el", "la", "?"],
                pattern="glue0 glue1 glue2 slot0 glue4"
            ),
            
            # Complex sentences with subordinate clauses (que, porque, cuando, si)
            Template(
                name="NOUN_VERB_QUE_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "que", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            Template(
                name="NOUN_VERB_PORQUE_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "porque", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            Template(
                name="NOUN_VERB_CUANDO_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "cuando", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            Template(
                name="SI_NOUN_VERB_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["si", "el", "la", "el", "la"],
                pattern="glue0 glue1 slot0 slot1 glue3 slot2 slot3"
            ),
            Template(
                name="NOUN_VERB_AUNQUE_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "aunque", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            
            # Compound sentences (y, pero, o)
            Template(
                name="NOUN_VERB_Y_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "y", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            Template(
                name="NOUN_VERB_PERO_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "pero", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            Template(
                name="NOUN_VERB_O_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "o", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3"
            ),
            
            # Compound-complex sentences
            Template(
                name="NOUN_VERB_Y_NOUN_VERB_PORQUE_NOUN_ADJ",
                slots=["NOUN", "VERB", "NOUN", "VERB", "NOUN", "ADJ"],
                glue_words=["el", "la", "y", "el", "la", "porque", "el", "la", "es"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3 glue5 glue6 slot4 glue8 slot5"
            ),
            Template(
                name="SI_NOUN_VERB_NOUN_VERB_PERO_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["si", "el", "la", "el", "la", "pero", "el", "la"],
                pattern="glue0 glue1 slot0 slot1 glue3 slot2 slot3 glue5 glue6 slot4 slot5"
            ),
            Template(
                name="NOUN_VERB_CUANDO_NOUN_VERB_Y_NOUN_VERB",
                slots=["NOUN", "VERB", "NOUN", "VERB", "NOUN", "VERB"],
                glue_words=["el", "la", "cuando", "el", "la", "y", "el", "la"],
                pattern="glue0 slot0 slot1 glue2 glue3 slot2 slot3 glue5 glue6 slot4 slot5"
            ),
            
            # Imperative patterns (commands)
            Template(
                name="IMP_VERB",
                slots=["VERB"],
                glue_words=["¡", "!"],
                pattern="glue0 slot0 glue1",
                mood="Imp"
            ),
            Template(
                name="IMP_VERB_NOUN",
                slots=["VERB", "NOUN"],
                glue_words=["¡", "el", "la", "!"],
                pattern="glue0 slot0 glue1 slot1 glue3",
                mood="Imp"
            ),
            Template(
                name="IMP_VERB_ADV",
                slots=["VERB", "ADV"],
                glue_words=["¡", "!"],
                pattern="glue0 slot0 slot1 glue1",
                mood="Imp"
            ),
            Template(
                name="IMP_NO_VERB",
                slots=["VERB"],
                glue_words=["¡", "no", "!"],
                pattern="glue0 glue1 slot0 glue2",
                mood="Imp"
            ),
            Template(
                name="IMP_VERB_PREP_NOUN",
                slots=["VERB", "ADP", "NOUN"],
                glue_words=["¡", "el", "la", "!"],
                pattern="glue0 slot0 slot1 glue1 slot2 glue3",
                mood="Imp"
            ),
            
            # Subjunctive patterns (wishes, doubts, hypotheticals)
            Template(
                name="ESPERO_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["espero", "que", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="QUIERO_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["quiero", "que", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="ES_IMPORTANTE_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["es", "importante", "que", "el", "la"],
                pattern="glue0 glue1 glue2 glue3 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="OJALA_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["ojalá", "que", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="DUDO_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["dudo", "que", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="ES_POSIBLE_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["es", "posible", "que", "el", "la"],
                pattern="glue0 glue1 glue2 glue3 slot0 slot1",
                mood="Sub"
            ),
            Template(
                name="ESPERO_QUE_NOUN_VERB_NOUN",
                slots=["NOUN", "VERB", "NOUN"],
                glue_words=["espero", "que", "el", "la", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1 glue4 slot2",
                mood="Sub"
            ),
            Template(
                name="QUIERO_QUE_NOUN_VERB_ADJ",
                slots=["NOUN", "VERB", "ADJ"],
                glue_words=["quiero", "que", "el", "la"],
                pattern="glue0 glue1 glue2 slot0 slot1 slot2",
                mood="Sub"
            ),
            Template(
                name="NO_CREO_QUE_NOUN_VERB",
                slots=["NOUN", "VERB"],
                glue_words=["no", "creo", "que", "el", "la"],
                pattern="glue0 glue1 glue2 glue3 slot0 slot1",
                mood="Sub"
            ),
        ]
        
        return templates
    
    def get_templates(self, max_words: int = 10) -> List[Template]:
        """
        Get templates that don't exceed max_words.
        
        Args:
            max_words: Maximum number of words allowed
            
        Returns:
            List of templates within word limit
        """
        return [t for t in self.templates if t.max_words() <= max_words]
    
    def get_templates_using_pos(self, pos_tags: List[str], max_words: int = 10) -> List[Template]:
        """
        Get templates that use specific POS tags.
        
        Args:
            pos_tags: List of POS tags to match
            max_words: Maximum number of words allowed
            
        Returns:
            List of matching templates
        """
        matching = []
        for template in self.templates:
            if template.max_words() <= max_words:
                # Check if template uses any of the specified POS tags
                if any(pos in template.slots for pos in pos_tags):
                    matching.append(template)
        return matching
    
    def get_all_templates(self) -> List[Template]:
        """Get all templates."""
        return self.templates
