"""Word-level adversarial attacks for NLP models."""
import random
import re
from typing import Dict, Any, List, Tuple, Optional, Set
import string

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from app.tests.nlp.adversarial.base import AdversarialAttack


class WordLevelAttack(AdversarialAttack):
    """Base class for word-level attacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Default configuration
        self.default_config = {
            "intensity": 0.1,  # Percentage of words to perturb
            "max_perturbations": 3,  # Maximum number of perturbations per text
            "min_word_length": 4,  # Minimum word length to consider for perturbation
            "preserve_pos": True,  # Whether to preserve part of speech
        }
        
        # Merge with provided config
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
    
    async def perturb(self, text: str) -> str:
        """Apply a word-level perturbation to the text."""
        words = text.split()
        
        # Filter out short words and words with non-alpha characters
        eligible_indices = [
            i for i, word in enumerate(words) 
            if len(word) >= self.config["min_word_length"] and word.isalpha()
        ]
        
        if not eligible_indices:
            return text  # No eligible words to modify
        
        # Determine how many words to perturb
        num_perturbations = min(
            max(1, int(len(eligible_indices) * self.config["intensity"])),
            self.config["max_perturbations"],
            len(eligible_indices)
        )
        
        # Select words to perturb
        indices_to_perturb = random.sample(eligible_indices, num_perturbations)
        
        # Perturb each selected word
        for idx in indices_to_perturb:
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


class SynonymAttack(WordLevelAttack):
    """Attack that replaces words with synonyms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "SynonymAttack"
        
        # Initialize WordNet if NLTK is available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
        
        # Simple synonym dictionary as fallback
        self.synonym_dict = {
            "happy": ["glad", "joyful", "pleased", "delighted"],
            "sad": ["unhappy", "sorrowful", "depressed", "gloomy"],
            "good": ["great", "excellent", "fine", "superb"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "gigantic"],
            "small": ["tiny", "little", "miniature", "minute"],
            "smart": ["intelligent", "clever", "bright", "brilliant"],
            "stupid": ["dumb", "foolish", "idiotic", "moronic"],
            "beautiful": ["pretty", "lovely", "gorgeous", "attractive"],
            "ugly": ["unattractive", "hideous", "unsightly", "plain"],
            "fast": ["quick", "speedy", "swift", "rapid"],
            "slow": ["sluggish", "leisurely", "gradual", "unhurried"],
            "strong": ["powerful", "mighty", "sturdy", "tough"],
            "weak": ["feeble", "frail", "fragile", "delicate"],
            "hot": ["warm", "heated", "scorching", "burning"],
            "cold": ["cool", "chilly", "freezing", "icy"],
            "rich": ["wealthy", "affluent", "prosperous", "opulent"],
            "poor": ["impoverished", "destitute", "needy", "indigent"],
            "old": ["aged", "elderly", "ancient", "senior"],
            "new": ["recent", "modern", "fresh", "novel"],
            "easy": ["simple", "straightforward", "effortless", "uncomplicated"],
            "hard": ["difficult", "challenging", "arduous", "tough"],
            "loud": ["noisy", "boisterous", "deafening", "thunderous"],
            "quiet": ["silent", "hushed", "inaudible", "muted"],
            "clean": ["spotless", "pristine", "immaculate", "unsoiled"],
            "dirty": ["soiled", "filthy", "grimy", "unclean"],
            "important": ["significant", "crucial", "essential", "vital"],
            "unimportant": ["insignificant", "trivial", "minor", "negligible"],
            "interesting": ["fascinating", "engaging", "compelling", "captivating"],
            "boring": ["dull", "tedious", "monotonous", "uninteresting"],
            "city": ["metropolis", "municipality", "urban area", "town"],
            "country": ["nation", "state", "land", "territory"],
            "house": ["home", "residence", "dwelling", "abode"],
            "car": ["automobile", "vehicle", "sedan", "motor"],
            "job": ["occupation", "profession", "career", "vocation"],
            "money": ["cash", "currency", "funds", "capital"],
            "friend": ["companion", "ally", "associate", "comrade"],
            "enemy": ["foe", "adversary", "opponent", "rival"],
            "teacher": ["instructor", "educator", "mentor", "tutor"],
            "student": ["pupil", "scholar", "learner", "apprentice"],
            "doctor": ["physician", "practitioner", "medic", "clinician"],
            "lawyer": ["attorney", "advocate", "counsel", "solicitor"],
            "book": ["volume", "publication", "tome", "text"],
            "movie": ["film", "picture", "motion picture", "feature"],
            "music": ["melody", "harmony", "rhythm", "sound"],
            "food": ["nourishment", "sustenance", "provisions", "fare"],
            "water": ["liquid", "fluid", "aqua", "H2O"],
        }
    
    def _perturb_word(self, word: str) -> str:
        """Replace a word with a synonym."""
        # Try to get synonyms from WordNet if available
        synonyms = self._get_synonyms(word)
        
        # If no synonyms found or not using WordNet, use fallback dictionary
        if not synonyms:
            word_lower = word.lower()
            if word_lower in self.synonym_dict:
                synonyms = self.synonym_dict[word_lower]
        
        if not synonyms:
            return word  # No synonyms found
        
        # Select a random synonym
        synonym = random.choice(synonyms)
        
        # Preserve capitalization
        if word.istitle():
            synonym = synonym.capitalize()
        elif word.isupper():
            synonym = synonym.upper()
            
        return synonym
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if not NLTK_AVAILABLE:
            return []
        
        word = word.lower()
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word and synonym not in synonyms and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def get_description(self) -> str:
        return "Replaces words with synonyms or semantically similar words"


class WordScrambleAttack(WordLevelAttack):
    """Attack that scrambles the internal letters of words."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "WordScrambleAttack"
    
    def _perturb_word(self, word: str) -> str:
        """Scramble the internal letters of a word, keeping the first and last letters fixed."""
        if len(word) <= 3:
            return word  # Too short to scramble meaningfully
        
        # Extract first and last letter
        first = word[0]
        last = word[-1]
        middle = word[1:-1]
        
        # Check if middle has meaningful content to scramble
        if len(middle) <= 1 or not any(c.isalpha() for c in middle):
            return word
        
        # Convert middle to list, shuffle, and join
        middle_list = list(middle)
        random.shuffle(middle_list)
        scrambled_middle = ''.join(middle_list)
        
        return first + scrambled_middle + last
    
    def get_description(self) -> str:
        return "Scrambles the internal letters of words while keeping the first and last letters fixed"


class WordInsertDeleteAttack(WordLevelAttack):
    """Attack that inserts or deletes words from the text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "WordInsertDeleteAttack"
        
        # Common filler words to insert
        self.filler_words = [
            "basically", "actually", "literally", "really", "very", 
            "quite", "simply", "just", "sort of", "kind of", "rather",
            "somewhat", "merely", "fairly", "pretty", "so", "totally",
            "absolutely", "completely", "indeed", "certainly", "surely",
            "definitely", "obviously", "apparently", "perhaps", "maybe",
            "possibly", "probably", "generally", "usually", "typically",
            "occasionally", "sometimes", "often", "frequently", "rarely"
        ]
    
    async def perturb(self, text: str) -> str:
        """Either insert or delete a word from the text."""
        if random.random() < 0.7:  # 70% chance to insert, 30% to delete
            return self._insert_word(text)
        else:
            return self._delete_word(text)
    
    def _insert_word(self, text: str) -> str:
        """Insert a filler word into the text."""
        words = text.split()
        if not words:
            return text
        
        # Choose a random position to insert the word
        insert_pos = random.randint(0, len(words))
        
        # Choose a random filler word
        filler = random.choice(self.filler_words)
        
        # Insert the word
        words.insert(insert_pos, filler)
        
        return " ".join(words)
    
    def _delete_word(self, text: str) -> str:
        """Delete a less important word from the text."""
        words = text.split()
        if len(words) <= 3:  # Don't delete if too few words
            return text
        
        # Common words that can often be removed without changing meaning significantly
        stopwords = {
            "a", "an", "the", "this", "that", "these", "those", 
            "and", "but", "or", "so", "yet", "for", "nor", 
            "on", "in", "at", "by", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "very", "quite", "rather", "somewhat", "fairly"
        }
        
        # Find stopwords in the text
        stopword_indices = [i for i, word in enumerate(words) if word.lower() in stopwords]
        
        if not stopword_indices:  # No stopwords found
            return text
        
        # Choose a random stopword to delete
        del_idx = random.choice(stopword_indices)
        words.pop(del_idx)
        
        return " ".join(words)
    
    def get_description(self) -> str:
        return "Inserts filler words or deletes less important words while preserving the main meaning" 