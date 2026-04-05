"""Content DNA Engine — 14-dimension voice fingerprinting for style consistency.

FIX: Profiles persist to SQLite and reload on startup (was in-memory only).
"""
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import textstat

from contentai_pro.core.config import settings

logger = logging.getLogger("contentai")


# The 14 DNA dimensions
DNA_DIMENSIONS = [
    "sentence_length_avg",    # Avg words per sentence
    "sentence_variance",      # Std dev of sentence lengths
    "vocabulary_tier",        # % advanced/rare words
    "passive_voice_ratio",    # % passive constructions
    "question_frequency",     # Questions per 100 sentences
    "metaphor_density",       # Figurative language per 1000 words
    "technical_depth",        # Domain jargon density
    "paragraph_rhythm",       # Avg sentences per paragraph
    "transition_density",     # Transition words per 100 words
    "contraction_ratio",      # Contracted vs full forms
    "first_person_usage",     # I/we/my/our frequency
    "exclamation_energy",     # Exclamation marks per 1000 words
    "list_structure_ratio",   # Bullet/numbered lists vs prose
    "opening_hook_style",     # 0=stat, 0.33=question, 0.66=story, 1=statement
]


@dataclass
class DNAProfile:
    name: str
    fingerprint: Dict[str, float] = field(default_factory=dict)
    samples_count: int = 0


class DNAEngine:
    """Analyzes writing samples to build a voice fingerprint, then scores new content for consistency."""

    def __init__(self):
        self.profiles: Dict[str, DNAProfile] = {}

    async def load_from_db(self, db_instance) -> int:
        """Load all persisted DNA profiles from database on startup.
        Returns the number of profiles loaded."""
        try:
            if not db_instance._conn:
                return 0
            cursor = await db_instance._conn.execute(
                "SELECT name, fingerprint, samples_count FROM dna_profiles ORDER BY rowid DESC"
            )
            rows = await cursor.fetchall()
            loaded = 0
            for row in rows:
                name = row[0] if isinstance(row, tuple) else row["name"]
                fingerprint_raw = row[1] if isinstance(row, tuple) else row["fingerprint"]
                samples = row[2] if isinstance(row, tuple) else row["samples_count"]
                try:
                    fp = json.loads(fingerprint_raw) if isinstance(fingerprint_raw, str) else fingerprint_raw
                    # Only load if not already in memory (in-memory takes precedence)
                    if name not in self.profiles:
                        self.profiles[name] = DNAProfile(
                            name=name, fingerprint=fp, samples_count=samples
                        )
                        loaded += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping corrupt DNA profile '{name}': {e}")
            logger.info(f"Loaded {loaded} DNA profiles from database")
            return loaded
        except Exception as e:
            logger.error(f"Failed to load DNA profiles from database: {e}")
            return 0

    def analyze_sample(self, text: str) -> Dict[str, float]:
        """Extract 14-dimension fingerprint from a single text sample."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        words = text.split()
        word_count = len(words)
        sent_count = max(len(sentences), 1)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Sentence lengths
        sent_lens = [len(s.split()) for s in sentences]
        avg_sent = sum(sent_lens) / max(len(sent_lens), 1)
        variance = math.sqrt(sum((sent_len - avg_sent) ** 2 for sent_len in sent_lens) / max(len(sent_lens), 1)) if sent_lens else 0

        # Vocabulary tier (simple heuristic: words > 8 chars)
        advanced = sum(1 for w in words if len(w) > 8) / max(word_count, 1)

        # Passive voice (crude: "was/were/been + past participle pattern")
        passive_patterns = len(re.findall(r'\b(was|were|been|being|is|are)\s+\w+ed\b', text, re.I))
        passive_ratio = passive_patterns / max(sent_count, 1)

        # Questions
        questions = text.count('?')
        question_freq = (questions / max(sent_count, 1)) * 100

        # Metaphor density (heuristic: "like a", "as if", "metaphorically")
        metaphor_markers = len(re.findall(r'\blike a\b|\bas if\b|\bmetaphor\w*\b|\banalog\w*\b', text, re.I))
        metaphor_density = (metaphor_markers / max(word_count, 1)) * 1000

        # Technical depth (words with mixed case or digits — no regex to avoid ReDoS)
        tech_words = sum(
            1 for w in words
            if any(c.isdigit() for c in w)
            or (any(c.isupper() for c in w) and any(c.islower() for c in w))
        )
        tech_depth = tech_words / max(word_count, 1)

        # Paragraph rhythm
        sents_per_para = [len(re.split(r'[.!?]+', p)) for p in paragraphs] if paragraphs else [0]
        para_rhythm = sum(sents_per_para) / max(len(sents_per_para), 1)

        # Transition words
        transitions = len(re.findall(
            r'\b(however|therefore|moreover|furthermore|meanwhile|consequently|'
            r'nevertheless|additionally|specifically|ultimately|notably)\b', text, re.I
        ))
        transition_density = (transitions / max(word_count, 1)) * 100

        # Contractions
        contractions = len(re.findall(r"\b\w+'[a-z]+\b", text))
        contraction_ratio = contractions / max(word_count, 1)

        # First person
        first_person = len(re.findall(r'\b(I|we|my|our|me|us)\b', text))
        first_person_ratio = first_person / max(word_count, 1)

        # Exclamations
        exclamations = text.count('!')
        exclamation_energy = (exclamations / max(word_count, 1)) * 1000

        # List structures
        list_items = len(re.findall(r'^\s*[-*•]\s|^\s*\d+[.)]\s', text, re.M))
        list_ratio = list_items / max(sent_count, 1)

        # Opening hook style
        first_sent = sentences[0] if sentences else ""
        if re.search(r'\d+%|\d+\.\d+|\$\d+', first_sent):
            hook_style = 0.0  # stat
        elif '?' in first_sent:
            hook_style = 0.33  # question
        elif re.search(r'\bonce\b|\bwhen I\b|\bimagine\b|\bpicture\b', first_sent, re.I):
            hook_style = 0.66  # story
        else:
            hook_style = 1.0  # statement

        return {
            "sentence_length_avg": round(avg_sent, 2),
            "sentence_variance": round(variance, 2),
            "vocabulary_tier": round(advanced, 4),
            "passive_voice_ratio": round(passive_ratio, 4),
            "question_frequency": round(question_freq, 2),
            "metaphor_density": round(metaphor_density, 2),
            "technical_depth": round(tech_depth, 4),
            "paragraph_rhythm": round(para_rhythm, 2),
            "transition_density": round(transition_density, 2),
            "contraction_ratio": round(contraction_ratio, 4),
            "first_person_usage": round(first_person_ratio, 4),
            "exclamation_energy": round(exclamation_energy, 2),
            "list_structure_ratio": round(list_ratio, 4),
            "opening_hook_style": round(hook_style, 2),
        }

    def calibrate(self, name: str, samples: List[str]) -> DNAProfile:
        """Build a DNA profile from multiple writing samples (min 3 recommended)."""
        if len(samples) < settings.DNA_SAMPLE_MIN:
            raise ValueError(f"Need at least {settings.DNA_SAMPLE_MIN} samples, got {len(samples)}")

        fingerprints = [self.analyze_sample(s) for s in samples]

        # Average across all samples
        avg_fp = {}
        for dim in DNA_DIMENSIONS:
            values = [fp[dim] for fp in fingerprints]
            avg_fp[dim] = round(sum(values) / len(values), 4)

        profile = DNAProfile(name=name, fingerprint=avg_fp, samples_count=len(samples))
        self.profiles[name] = profile
        return profile

    def score(self, text: str, profile_name: str) -> Dict:
        """Score how closely a text matches a DNA profile. Returns 0-100."""
        if profile_name not in self.profiles:
            return {"error": f"Profile '{profile_name}' not found", "score": 0}

        profile = self.profiles[profile_name]
        current = self.analyze_sample(text)

        # Compute per-dimension similarity (1 - normalized distance)
        dim_scores = {}
        for dim in DNA_DIMENSIONS:
            target = profile.fingerprint.get(dim, 0)
            actual = current.get(dim, 0)
            max_val = max(abs(target), abs(actual), 1e-6)
            similarity = 1.0 - min(abs(target - actual) / max_val, 1.0)
            dim_scores[dim] = round(similarity * 100, 1)

        overall = round(sum(dim_scores.values()) / len(dim_scores), 1)

        return {
            "overall_score": overall,
            "dimension_scores": dim_scores,
            "profile_name": profile_name,
            "samples_analyzed": profile.samples_count,
            "current_fingerprint": current,
            "target_fingerprint": profile.fingerprint,
        }

    def get_profile_summary(self, profile_name: str) -> Optional[str]:
        """Generate a natural-language description of a DNA profile for agent prompts."""
        if profile_name not in self.profiles:
            return None
        fp = self.profiles[profile_name].fingerprint
        parts = []
        if fp.get("sentence_length_avg", 0) > 20:
            parts.append("long, complex sentences")
        elif fp.get("sentence_length_avg", 0) < 12:
            parts.append("short, punchy sentences")
        if fp.get("contraction_ratio", 0) > 0.02:
            parts.append("conversational tone with contractions")
        if fp.get("technical_depth", 0) > 0.1:
            parts.append("high technical vocabulary density")
        if fp.get("first_person_usage", 0) > 0.02:
            parts.append("first-person perspective")
        if fp.get("metaphor_density", 0) > 2:
            parts.append("rich use of metaphors and analogies")
        if fp.get("question_frequency", 0) > 10:
            parts.append("frequent rhetorical questions")
        return f"Voice profile: {', '.join(parts)}." if parts else "Neutral professional voice."

    # ------------------------------------------------------------------
    # Algorithmic fingerprinting using textstat (no LLM required)
    # ------------------------------------------------------------------

    def compute_fingerprint(self, text: str) -> Dict[str, float]:
        """Compute a 14-dimension voice fingerprint using algorithmic analysis.

        Uses textstat for readability metrics instead of LLM calls.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()

        return {
            "sentence_length_avg": round(sum(len(s.split()) for s in sentences) / max(len(sentences), 1), 2),
            "sentence_variance": round(self._variance([len(s.split()) for s in sentences]), 2),
            "vocabulary_tier": round(textstat.difficult_words(text) / max(len(words), 1), 4),
            "passive_voice_ratio": round(self._count_passive(text) / max(len(sentences), 1), 4),
            "question_frequency": round(text.count('?') / max(len(sentences), 1) * 100, 2),
            "metaphor_density": round(len(re.findall(r'\b(like|as if|as though)\b', text.lower())) / max(len(words), 1) * 1000, 2),
            "technical_depth": round(textstat.avg_syllables_per_word(text), 4),
            "paragraph_rhythm": round(len(sentences) / max(text.count('\n\n') + 1, 1), 2),
            "transition_density": round(self._count_transitions(text) / max(len(words), 1) * 100, 2),
            "contraction_ratio": round(len(re.findall(r"\b\w+'\w+\b", text)) / max(len(words), 1), 4),
            "first_person_usage": round(len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', text, re.I)) / max(len(words), 1), 4),
            "exclamation_energy": round(text.count('!') / max(len(words), 1) * 1000, 2),
            "list_structure_ratio": round(len(re.findall(r'^\s*[-*•\d]+[.)]?\s', text, re.MULTILINE)) / max(len(sentences), 1), 4),
            "opening_hook_style": round(self._score_opening_hook(sentences[0] if sentences else ""), 2),
        }

    def _variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _count_passive(self, text: str) -> int:
        return len(re.findall(r'\b(was|were|been|being|is|are)\s+\w+ed\b', text.lower()))

    def _count_transitions(self, text: str) -> int:
        transitions = r'\b(however|therefore|moreover|furthermore|additionally|consequently|nevertheless|thus|hence|accordingly)\b'
        return len(re.findall(transitions, text.lower()))

    def _score_opening_hook(self, first_sentence: str) -> float:
        score = 0.5  # baseline
        if first_sentence.endswith('?'):
            score += 0.2  # question hook
        if re.match(r'^(imagine|picture|consider|what if)', first_sentence.lower()):
            score += 0.2  # imperative/hypothetical hook
        if len(first_sentence.split()) < 10:
            score += 0.1  # punchy short opener
        return min(score, 1.0)


# Singleton
dna_engine = DNAEngine()
