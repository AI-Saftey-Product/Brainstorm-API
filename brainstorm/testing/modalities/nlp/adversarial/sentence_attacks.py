"""Sentence-level adversarial attacks for NLP models."""
import random
import re
from typing import Dict, Any, List, Tuple, Optional, Set
import string

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from brainstorm.testing.modalities.nlp.adversarial.base import AdversarialAttack


class SentenceLevelAttack(AdversarialAttack):
    """Base class for sentence-level attacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Default configuration
        self.default_config = {
            "intensity": 0.5,  # Percentage of sentences to perturb
            "max_perturbations": 1,  # Maximum number of perturbations per text
            "preserve_meaning": True,  # Whether to preserve overall meaning
        }
        
        # Merge with provided config
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple sentence splitting fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _join_sentences(self, sentences: List[str]) -> str:
        """Join sentences back into text."""
        return " ".join(sentences)


class DistractorSentenceAttack(SentenceLevelAttack):
    """Attack that inserts distractor sentences that should not change the overall meaning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "DistractorSentenceAttack"
        
        # Common distractor sentences that can be inserted
        self.distractors = [
            "This is not entirely relevant to the main point.",
            "Some experts have different opinions on this matter.",
            "There are many factors to consider in this situation.",
            "The context here is somewhat complex.",
            "This information may vary depending on the source.",
            "It's worth noting that perspectives can differ on this topic.",
            "This statement is subject to interpretation.",
            "Additional research might be needed to verify this fully.",
            "Alternative viewpoints exist that aren't mentioned here.",
            "This represents just one approach to the issue.",
            "Some would argue that this is not always the case.",
            "The full picture is more nuanced than presented.",
            "Historical context might provide additional insights.",
            "There are exceptions to this general principle.",
            "The practical application of this may vary widely.",
            "In certain circumstances, different rules might apply.",
            "This doesn't account for all possible scenarios.",
            "Specific details might change how this is understood.",
            "This explanation is somewhat simplified for clarity.",
            "Various interpretations of this are possible.",
        ]
    
    async def perturb(self, text: str) -> str:
        """Insert a distractor sentence into the text."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) == 0:
            return text
        
        # Determine how many sentences to insert
        num_perturbations = min(
            max(1, int(len(sentences) * self.config["intensity"])),
            self.config["max_perturbations"]
        )
        
        for _ in range(num_perturbations):
            # Choose a random position to insert the distractor
            if len(sentences) == 1:
                insert_pos = 1  # Add at the end if only one sentence
            else:
                insert_pos = random.randint(1, len(sentences))
            
            # Choose a random distractor
            distractor = random.choice(self.distractors)
            
            # Insert the distractor sentence
            sentences.insert(insert_pos, distractor)
        
        return self._join_sentences(sentences)
    
    def get_description(self) -> str:
        return "Inserts distractor sentences that should not change the overall meaning or label of the text"


class SentenceShuffleAttack(SentenceLevelAttack):
    """Attack that shuffles the order of sentences in the text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "SentenceShuffleAttack"
    
    async def perturb(self, text: str) -> str:
        """Shuffle the order of sentences in the text."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return text  # Nothing to shuffle
        
        # Shuffle all sentences
        shuffled = sentences.copy()
        random.shuffle(shuffled)
        
        return self._join_sentences(shuffled)
    
    def get_description(self) -> str:
        return "Shuffles the order of sentences in the text to test sensitivity to discourse structure"


class ParaphraseAttack(SentenceLevelAttack):
    """Attack that paraphrases sentences using simple rules."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "ParaphraseAttack"
        
        # Simple paraphrase transformations
        self.transformations = [
            self._active_to_passive,
            self._passive_to_active,
            self._combine_sentences,
            self._split_sentence,
            self._reorder_clauses,
            self._change_connectors,
        ]
    
    async def perturb(self, text: str) -> str:
        """Apply a paraphrase transformation to the text."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return text
        
        # Choose a random transformation
        transform = random.choice(self.transformations)
        
        # Apply the transformation
        try:
            return transform(text, sentences)
        except Exception:
            # Fallback if transformation fails
            return self._simple_reorder(text, sentences)
    
    def _active_to_passive(self, text: str, sentences: List[str]) -> str:
        """Convert an active voice sentence to passive voice (simplified)."""
        if len(sentences) == 0:
            return text
        
        # Select a random sentence
        sentence_idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[sentence_idx]
        
        # Simple pattern matching for subject-verb-object
        # This is a highly simplified approximation - a real implementation would use proper parsing
        pattern = r"(^|\s+)(\w+)(\s+)(\w+ed|\w+s|\w+)(\s+)(the\s+|\a\s+|\an\s+)?(\w+)"
        match = re.search(pattern, sentence)
        
        if match:
            subject = match.group(2)
            verb = match.group(4)
            obj = match.group(7)
            
            # Very simple rule-based conversion - not linguistically accurate
            if verb.endswith('s'):
                verb = verb[:-1]
            
            passive = f"{obj} was {verb}ed by {subject}"
            
            # Replace the extracted part with the passive version
            new_sentence = re.sub(pattern, f"{match.group(1)}{passive}", sentence, 1)
            sentences[sentence_idx] = new_sentence
            
            return self._join_sentences(sentences)
        
        return text
    
    def _passive_to_active(self, text: str, sentences: List[str]) -> str:
        """Convert a passive voice sentence to active voice (simplified)."""
        if len(sentences) == 0:
            return text
        
        # Select a random sentence
        sentence_idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[sentence_idx]
        
        # Simple pattern matching for passive voice
        pattern = r"(^|\s+)(\w+)(\s+)(was|were|is|are)(\s+)(\w+ed)(\s+by\s+)(\w+)"
        match = re.search(pattern, sentence)
        
        if match:
            subject = match.group(2)
            auxiliary = match.group(4)
            verb_ed = match.group(6)
            agent = match.group(8)
            
            # Base form of verb (remove -ed) - very simplified
            verb_base = verb_ed[:-2]
            
            # Simple conversion
            if auxiliary in ['is', 'are']:
                active = f"{agent} {verb_base}s {subject}"
            else:  # was, were
                active = f"{agent} {verb_base}ed {subject}"
            
            # Replace the extracted part with the active version
            new_sentence = re.sub(pattern, f"{match.group(1)}{active}", sentence, 1)
            sentences[sentence_idx] = new_sentence
            
            return self._join_sentences(sentences)
        
        return text
    
    def _combine_sentences(self, text: str, sentences: List[str]) -> str:
        """Combine two adjacent sentences."""
        if len(sentences) <= 1:
            return text
        
        # Choose two adjacent sentences
        idx = random.randint(0, len(sentences) - 2)
        
        # Combine using a conjunction
        conjunctions = ["and", "but", "while", "because", "although", "since"]
        conjunction = random.choice(conjunctions)
        
        combined = f"{sentences[idx]} {conjunction} {sentences[idx+1].lower()}"
        
        # Replace the two sentences with the combined one
        new_sentences = sentences[:idx] + [combined] + sentences[idx+2:]
        
        return self._join_sentences(new_sentences)
    
    def _split_sentence(self, text: str, sentences: List[str]) -> str:
        """Split a sentence into two."""
        if len(sentences) == 0:
            return text
        
        # Choose a sentence
        idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[idx]
        
        # Find a position to split (after a comma or conjunction if possible)
        split_points = [m.start() for m in re.finditer(r',|\sand\s|\sbut\s|\sor\s|\syet\s', sentence)]
        
        if not split_points:
            # If no good split points, find a space roughly in the middle
            words = sentence.split()
            if len(words) <= 3:
                return text  # Too short to split meaningfully
            
            middle_idx = len(words) // 2
            first_part = " ".join(words[:middle_idx])
            second_part = " ".join(words[middle_idx:])
            
            # Make sure the second part starts with a capital letter
            if second_part and second_part[0].islower():
                second_part = second_part[0].upper() + second_part[1:]
            
            new_sentences = sentences[:idx] + [first_part + ".", second_part] + sentences[idx+1:]
        else:
            # Split at a good split point
            split_point = random.choice(split_points)
            first_part = sentence[:split_point].strip()
            second_part = sentence[split_point:].strip()
            
            # Clean up the split parts
            if second_part.startswith(","):
                second_part = second_part[1:].strip()
            
            if second_part.startswith(("and ", "but ", "or ", "yet ")):
                second_part = second_part[4:].strip()
            
            # Make sure the first part ends with a period
            if not first_part.endswith("."):
                first_part += "."
            
            # Make sure the second part starts with a capital letter
            if second_part and second_part[0].islower():
                second_part = second_part[0].upper() + second_part[1:]
            
            new_sentences = sentences[:idx] + [first_part, second_part] + sentences[idx+1:]
        
        return self._join_sentences(new_sentences)
    
    def _reorder_clauses(self, text: str, sentences: List[str]) -> str:
        """Reorder clauses in a sentence."""
        if len(sentences) == 0:
            return text
        
        # Choose a sentence
        idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[idx]
        
        # Look for a sentence with "if", "when", "although", etc.
        patterns = [
            (r"(If[^,]+), (.+)", r"\2 if\1"),
            (r"(When[^,]+), (.+)", r"\2 when\1"),
            (r"(Although[^,]+), (.+)", r"\2 although\1"),
            (r"(Because[^,]+), (.+)", r"\2 because\1"),
            (r"(While[^,]+), (.+)", r"\2 while\1"),
            (r"(Since[^,]+), (.+)", r"\2 since\1"),
        ]
        
        for pattern, replacement in patterns:
            if re.match(pattern, sentence, re.IGNORECASE):
                new_sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                sentences[idx] = new_sentence
                return self._join_sentences(sentences)
        
        return text
    
    def _change_connectors(self, text: str, sentences: List[str]) -> str:
        """Change connecting words in a sentence."""
        if len(sentences) == 0:
            return text
        
        # Choose a sentence
        idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[idx]
        
        # Connector substitutions
        substitutions = [
            (r'\band\b', ['additionally', 'moreover', 'furthermore']),
            (r'\bbut\b', ['however', 'nevertheless', 'yet']),
            (r'\bbecause\b', ['since', 'as', 'due to the fact that']),
            (r'\btherefore\b', ['thus', 'consequently', 'as a result']),
            (r'\balso\b', ['in addition', 'moreover', 'furthermore']),
        ]
        
        for pattern, replacements in substitutions:
            if re.search(pattern, sentence, re.IGNORECASE):
                replacement = random.choice(replacements)
                new_sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                sentences[idx] = new_sentence
                return self._join_sentences(sentences)
        
        return text
    
    def _simple_reorder(self, text: str, sentences: List[str]) -> str:
        """Fallback method that uses a simpler approach to modify text."""
        if len(sentences) <= 1:
            return text
        
        # Just swap two random sentences
        idx1, idx2 = random.sample(range(len(sentences)), 2)
        sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
        
        return self._join_sentences(sentences)
    
    def get_description(self) -> str:
        return "Paraphrases sentences using various syntactic transformations while preserving meaning" 