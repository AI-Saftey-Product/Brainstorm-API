"""Character-level adversarial attacks for NLP models."""
import random
import re
from typing import Dict, Any, List, Tuple, Optional
import string

from app.tests.nlp.adversarial.base import AdversarialAttack


class CharacterLevelAttack(AdversarialAttack):
    """Base class for character-level attacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Default configuration
        self.default_config = {
            "intensity": 0.1,  # Percentage of words to perturb
            "max_perturbations": 3,  # Maximum number of perturbations per text
            "preserve_word_boundaries": True,  # Whether to preserve word boundaries
        }
        
        # Merge with provided config
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
    
    async def perturb(self, text: str) -> str:
        """Apply a character-level perturbation to the text."""
        words = text.split()
        
        # Determine how many words to perturb
        num_words = len(words)
        num_perturbations = min(
            max(1, int(num_words * self.config["intensity"])),
            self.config["max_perturbations"]
        )
        
        # Select words to perturb
        words_to_perturb = random.sample(range(num_words), min(num_perturbations, num_words))
        
        # Perturb each selected word
        for idx in words_to_perturb:
            words[idx] = self._perturb_word(words[idx])
        
        return " ".join(words)
    
    def _perturb_word(self, word: str) -> str:
        """
        Perturb a single word. To be implemented by subclasses.
        
        Args:
            word: Word to perturb
            
        Returns:
            Perturbed word
        """
        return word


class TypoAttack(CharacterLevelAttack):
    """Attack that introduces typos by swapping, inserting, or deleting characters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "TypoAttack"
    
    def _perturb_word(self, word: str) -> str:
        """Add a typo to the word."""
        if len(word) <= 1:
            return word
        
        # Skip very short words and punctuation-only words
        if len(word) <= 2 or all(c in string.punctuation for c in word):
            return word
        
        # Choose a perturbation type
        perturbation_type = random.choice(["swap", "delete", "insert", "replace"])
        
        if perturbation_type == "swap" and len(word) >= 3:
            # Swap two adjacent characters (not first or last)
            idx = random.randint(1, len(word) - 2)
            return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
        
        elif perturbation_type == "delete":
            # Delete a random character (not first or last for readability)
            if len(word) <= 3:
                idx = random.randint(0, len(word) - 1)
            else:
                idx = random.randint(1, len(word) - 2)
            return word[:idx] + word[idx+1:]
        
        elif perturbation_type == "insert":
            # Insert a random character
            char = random.choice(string.ascii_lowercase)
            idx = random.randint(1, len(word) - 1)
            return word[:idx] + char + word[idx:]
        
        elif perturbation_type == "replace":
            # Replace a character with a similar one
            if len(word) <= 2:
                idx = random.randint(0, len(word) - 1)
            else:
                idx = random.randint(1, len(word) - 1)  # Avoid first char for readability
            
            char = word[idx]
            similar_chars = self._get_similar_chars(char)
            replacement = random.choice(similar_chars) if similar_chars else char
            return word[:idx] + replacement + word[idx+1:]
        
        return word
    
    def _get_similar_chars(self, char: str) -> List[str]:
        """Get characters similar to the input character (keyboard neighbors)."""
        keyboard_neighbors = {
            'a': ['s', 'q', 'z', 'w'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p', ';'],
            'm': ['n', 'j', 'k', ','],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l', ';', '['],
            'q': ['1', 'w', 'a'],
            'r': ['e', 'd', 'f', 't'],
            's': ['a', 'z', 'x', 'd', 'e', 'w'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x']
        }
        
        char = char.lower()
        if char in keyboard_neighbors:
            return keyboard_neighbors[char]
        return []
    
    def get_description(self) -> str:
        return "Introduces common typographical errors through character swapping, deletion, insertion, or replacement"


class HomoglyphAttack(CharacterLevelAttack):
    """Attack that replaces characters with visually similar ones (homoglyphs)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "HomoglyphAttack"
        
        # Map of original characters to homoglyphs
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α', 'ａ'],  # Cyrillic 'а', Latin 'ɑ', Greek 'α', fullwidth 'a'
            'b': ['Ь', 'ｂ'],  # Cyrillic 'Ь', fullwidth 'b'
            'c': ['с', 'ϲ', 'ｃ'],  # Cyrillic 'с', Greek 'ϲ', fullwidth 'c'
            'd': ['ԁ', 'ｄ'],  # Cyrillic 'ԁ', fullwidth 'd'
            'e': ['е', 'ε', 'ｅ'],  # Cyrillic 'е', Greek 'ε', fullwidth 'e'
            'g': ['ɡ', 'ｇ'],  # Latin 'ɡ', fullwidth 'g'
            'h': ['һ', 'ｈ'],  # Cyrillic 'һ', fullwidth 'h'
            'i': ['і', 'ｉ'],  # Cyrillic 'і', fullwidth 'i'
            'j': ['ј', 'ｊ'],  # Cyrillic 'ј', fullwidth 'j'
            'k': ['ｋ'],  # fullwidth 'k'
            'l': ['ӏ', 'ｌ'],  # Cyrillic 'ӏ', fullwidth 'l'
            'm': ['ｍ'],  # fullwidth 'm'
            'n': ['ｎ'],  # fullwidth 'n'
            'o': ['о', 'ο', 'ｏ'],  # Cyrillic 'о', Greek 'ο', fullwidth 'o'
            'p': ['р', 'ρ', 'ｐ'],  # Cyrillic 'р', Greek 'ρ', fullwidth 'p'
            'q': ['ｑ'],  # fullwidth 'q'
            'r': ['г', 'ｒ'],  # Cyrillic 'г', fullwidth 'r'
            's': ['ѕ', 'ｓ'],  # Cyrillic 'ѕ', fullwidth 's'
            't': ['т', 'ｔ'],  # Cyrillic 'т', fullwidth 't'
            'u': ['ս', 'ｕ'],  # Armenian 'ս', fullwidth 'u'
            'v': ['ѵ', 'ｖ'],  # Cyrillic 'ѵ', fullwidth 'v'
            'w': ['ԝ', 'ｗ'],  # Cyrillic 'ԝ', fullwidth 'w'
            'x': ['х', 'ｘ'],  # Cyrillic 'х', fullwidth 'x'
            'y': ['у', 'ｙ'],  # Cyrillic 'у', fullwidth 'y'
            'z': ['ｚ']  # fullwidth 'z'
        }
    
    def _perturb_word(self, word: str) -> str:
        """Replace a random character with a homoglyph."""
        if len(word) <= 1:
            return word
        
        # Find eligible characters (those with homoglyphs)
        eligible_indices = []
        for i, char in enumerate(word):
            if char.lower() in self.homoglyphs:
                eligible_indices.append(i)
        
        if not eligible_indices:
            return word
        
        # Select a random eligible character
        idx = random.choice(eligible_indices)
        char = word[idx].lower()
        
        # Check if we have homoglyphs for this character
        if char in self.homoglyphs and self.homoglyphs[char]:
            # Replace with a random homoglyph
            homoglyph = random.choice(self.homoglyphs[char])
            
            # Preserve case if possible
            if word[idx].isupper() and homoglyph.upper() != homoglyph:
                homoglyph = homoglyph.upper()
            
            return word[:idx] + homoglyph + word[idx+1:]
        
        return word
    
    def get_description(self) -> str:
        return "Replaces characters with visually similar Unicode characters that look identical but have different code points"


class PunctuationAttack(CharacterLevelAttack):
    """Attack that modifies punctuation and spaces in text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "PunctuationAttack"
    
    async def perturb(self, text: str) -> str:
        """Apply punctuation or space perturbations to the text."""
        # Choose a perturbation strategy
        strategy = random.choice([
            "add_space", "remove_space", "add_punctuation", "remove_punctuation"
        ])
        
        if strategy == "add_space":
            return self._add_random_space(text)
        elif strategy == "remove_space":
            return self._remove_random_space(text)
        elif strategy == "add_punctuation":
            return self._add_random_punctuation(text)
        elif strategy == "remove_punctuation":
            return self._remove_random_punctuation(text)
        
        return text
    
    def _add_random_space(self, text: str) -> str:
        """Add a space in the middle of a random word."""
        words = text.split()
        if not words:
            return text
        
        # Find words long enough to split
        eligible_words = [i for i, word in enumerate(words) if len(word) > 3]
        if not eligible_words:
            return text
        
        word_idx = random.choice(eligible_words)
        word = words[word_idx]
        
        # Insert space at a random position (not first or last character)
        pos = random.randint(1, len(word) - 1)
        new_word = word[:pos] + " " + word[pos:]
        words[word_idx] = new_word
        
        return " ".join(words)
    
    def _remove_random_space(self, text: str) -> str:
        """Remove a random space between words."""
        if " " not in text:
            return text
        
        # Find all spaces
        spaces = [m.start() for m in re.finditer(' ', text)]
        if not spaces:
            return text
        
        # Choose a random space to remove
        space_idx = random.choice(spaces)
        return text[:space_idx] + text[space_idx+1:]
    
    def _add_random_punctuation(self, text: str) -> str:
        """Add a random punctuation mark in the text."""
        if not text:
            return text
        
        punctuation = ['.', ',', ';', ':', '!', '?']
        punct = random.choice(punctuation)
        
        # Either add to a random position or between words
        if random.random() < 0.5:
            # Add to a random position (not first character)
            pos = random.randint(1, len(text) - 1)
            return text[:pos] + punct + text[pos:]
        else:
            # Add between words
            words = text.split()
            if len(words) <= 1:
                return text
            
            word_idx = random.randint(0, len(words) - 2)
            words[word_idx] = words[word_idx] + punct
            return " ".join(words)
    
    def _remove_random_punctuation(self, text: str) -> str:
        """Remove a random punctuation mark from the text."""
        # Find all punctuation
        punct_indices = [i for i, char in enumerate(text) if char in string.punctuation]
        if not punct_indices:
            return text
        
        # Choose a random punctuation to remove
        punct_idx = random.choice(punct_indices)
        return text[:punct_idx] + text[punct_idx+1:]
    
    def get_description(self) -> str:
        return "Modifies text by adding or removing spaces and punctuation"


class UnicodeAttack(CharacterLevelAttack):
    """Attack that uses various Unicode manipulation techniques including invisible characters,
    combining marks, extended confusables, and bidirectional text markers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "UnicodeAttack"
        
        # Different categories of Unicode manipulations
        self.invisible_chars = [
            '\u200B',  # Zero-width space
            '\u200C',  # Zero-width non-joiner
            '\u200D',  # Zero-width joiner
            '\u2060',  # Word joiner
            '\uFEFF'   # Zero-width non-breaking space
        ]
        
        self.combining_marks = [
            '\u0300',  # Combining grave accent
            '\u0301',  # Combining acute accent
            '\u0302',  # Combining circumflex
            '\u0308',  # Combining diaeresis
            '\u0327',  # Combining cedilla
            '\u0334',  # Combining tilde overlay
            '\u0336'   # Combining strike
        ]
        
        # Extended confusables beyond the current homoglyphs
        self.confusables = {
            'a': ['ą', 'ă', 'ã', 'ā', 'ȧ'],
            'e': ['ę', 'ė', 'ē', 'ě', 'è'],
            'i': ['ī', 'ĭ', 'ĩ', 'į', 'ì'],
            'o': ['ō', 'ŏ', 'õ', 'ő', 'ò'],
            'u': ['ū', 'ŭ', 'ũ', 'ų', 'ù'],
            'n': ['ń', 'ņ', 'ñ', 'ň', 'ǹ'],
            'y': ['ý', 'ỳ', 'ŷ', 'ÿ', 'ỹ'],
            's': ['ś', 'ş', 'š', 'ș', 'ṣ'],
            'l': ['ĺ', 'ļ', 'ľ', 'ł', 'ḷ'],
            'z': ['ź', 'ż', 'ž', 'ẑ', 'ẓ']
        }
        
        self.rtl_markers = [
            '\u202E',  # Right-to-left override
            '\u202D',  # Left-to-right override
            '\u202B',  # Right-to-left embedding
            '\u202A'   # Left-to-right embedding
        ]
        
        # Update config with specific settings
        self.default_config.update({
            "invisible_char_prob": 0.3,  # Probability of adding invisible characters
            "combining_mark_prob": 0.3,  # Probability of adding combining marks
            "confusable_prob": 0.4,      # Probability of using confusables
            "rtl_prob": 0.1              # Probability of adding RTL markers
        })
        
        if config:
            self.default_config.update(config)
    
    def _perturb_word(self, word: str) -> str:
        """Apply Unicode-based perturbation to a word."""
        if len(word) <= 1:
            return word
            
        # Choose attack strategy based on probabilities
        strategy = random.choices(
            [
                self._add_invisible_chars,
                self._add_combining_marks,
                self._replace_with_confusables,
                self._add_rtl_markers
            ],
            weights=[
                self.config["invisible_char_prob"],
                self.config["combining_mark_prob"],
                self.config["confusable_prob"],
                self.config["rtl_prob"]
            ]
        )[0]
        
        return strategy(word)
    
    def _add_invisible_chars(self, word: str) -> str:
        """Insert invisible characters between letters."""
        if len(word) <= 1:
            return word
        
        # Insert 1-3 invisible characters at random positions
        num_inserts = random.randint(1, min(3, len(word) - 1))
        positions = random.sample(range(1, len(word)), num_inserts)
        
        result = list(word)
        for pos in sorted(positions, reverse=True):
            char = random.choice(self.invisible_chars)
            result.insert(pos, char)
        
        return ''.join(result)
    
    def _add_combining_marks(self, word: str) -> str:
        """Add combining diacritical marks to letters."""
        result = []
        for char in word:
            result.append(char)
            if random.random() < 0.3:  # 30% chance per character
                num_marks = random.randint(1, 2)
                marks = random.sample(self.combining_marks, num_marks)
                result.extend(marks)
        return ''.join(result)
    
    def _replace_with_confusables(self, word: str) -> str:
        """Replace characters with confusable Unicode alternatives."""
        result = []
        for char in word:
            if char.lower() in self.confusables and random.random() < 0.4:
                confusable = random.choice(self.confusables[char.lower()])
                # Preserve case if possible
                if char.isupper():
                    confusable = confusable.upper()
                result.append(confusable)
            else:
                result.append(char)
        return ''.join(result)
    
    def _add_rtl_markers(self, word: str) -> str:
        """Add RTL/LTR markers to create bidirectional text."""
        if len(word) <= 2:
            return word
            
        # Add RTL marker in the middle of the word
        mid = len(word) // 2
        marker = random.choice(self.rtl_markers)
        return word[:mid] + marker + word[mid:]
    
    def get_description(self) -> str:
        return "Manipulates text using various Unicode techniques including invisible characters, combining marks, confusables, and bidirectional markers" 