"""Content DNA Engine — multi-layered voice fingerprinting for style consistency.

Layers
------
Macro DNA   : Persistent brand voice (consistent across all content types).
Micro DNA   : Format-specific patterns keyed by content_type.
Temporal DNA: Chronological snapshots for evolution/drift tracking.
Contextual DNA: Industry- or audience-specific adaptations.
"""
import re
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Tuple
from contentai_pro.core.config import settings

_EPSILON = 1e-9           # Guard against division by zero in drift computation
_MIN_SCORE_DIVISOR = 1e-6  # Minimum denominator for score similarity computation


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


class DNALayer(str, Enum):
    MACRO = "macro"          # Brand-wide voice constants
    MICRO = "micro"          # Format-specific (blog, thread, email …)
    TEMPORAL = "temporal"    # Historical snapshot for drift detection
    CONTEXTUAL = "contextual"  # Industry / audience adaptation


@dataclass
class DNAVersion:
    """Immutable snapshot of a fingerprint at a point in time."""
    version_id: str
    layer: DNALayer
    fingerprint: Dict[str, float]
    label: str = ""           # e.g. "v1", "blog_variant_A"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DriftAlert:
    profile_name: str
    dimension: str
    baseline: float
    current: float
    delta_pct: float          # % change vs baseline
    threshold_pct: float
    exceeded: bool


@dataclass
class DNAProfile:
    name: str
    fingerprint: Dict[str, float] = field(default_factory=dict)   # Macro baseline
    samples_count: int = 0
    # Multi-layer storage
    macro_dna: Dict[str, float] = field(default_factory=dict)
    micro_dna: Dict[str, Dict[str, float]] = field(default_factory=dict)   # keyed by content_type
    contextual_dna: Dict[str, Dict[str, float]] = field(default_factory=dict)  # keyed by context key
    versions: List[DNAVersion] = field(default_factory=list)
    drift_threshold_pct: float = 20.0  # Alert when dimension drifts > this %


class DNAEngine:
    """Analyzes writing samples to build a multi-layered voice fingerprint.

    Layers
    ------
    Macro  — brand-wide constants calibrated from all samples.
    Micro  — per-format patterns (blog, thread, email …).
    Temporal — chronological snapshots for drift detection.
    Contextual — industry/audience-specific adaptations.
    """

    def __init__(self):
        self.profiles: Dict[str, DNAProfile] = {}

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
        variance = math.sqrt(sum((l - avg_sent) ** 2 for l in sent_lens) / max(len(sent_lens), 1)) if sent_lens else 0

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

        # Technical depth (words with mixed case, numbers, or special chars)
        tech_words = sum(1 for w in words if re.search(r'[A-Z].*[a-z]|[a-z].*[A-Z]|\d', w))
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

        if name not in self.profiles:
            profile = DNAProfile(name=name, fingerprint=avg_fp, samples_count=len(samples))
        else:
            profile = self.profiles[name]
            profile.fingerprint = avg_fp
            profile.samples_count = len(samples)

        profile.macro_dna = avg_fp

        # Create initial temporal snapshot
        version = DNAVersion(
            version_id=str(uuid.uuid4()),
            layer=DNALayer.TEMPORAL,
            fingerprint=dict(avg_fp),
            label=f"v{len(profile.versions) + 1}",
        )
        profile.versions.append(version)

        self.profiles[name] = profile
        return profile

    def calibrate_layer(
        self,
        name: str,
        samples: List[str],
        layer: DNALayer,
        context_key: str = "",
    ) -> DNAProfile:
        """Calibrate a specific DNA layer for an existing profile.

        Parameters
        ----------
        name        : Profile name (must already exist or will be created).
        samples     : Writing samples representative of this layer.
        layer       : Which layer to calibrate.
        context_key : Required for MICRO (content_type) and CONTEXTUAL (context key).
        """
        if len(samples) < 1:
            raise ValueError("Need at least 1 sample to calibrate a layer.")

        fingerprints = [self.analyze_sample(s) for s in samples]
        avg_fp: Dict[str, float] = {}
        for dim in DNA_DIMENSIONS:
            values = [fp[dim] for fp in fingerprints]
            avg_fp[dim] = round(sum(values) / len(values), 4)

        if name not in self.profiles:
            self.profiles[name] = DNAProfile(name=name)
        profile = self.profiles[name]

        if layer == DNALayer.MACRO:
            profile.macro_dna = avg_fp
            profile.fingerprint = avg_fp
            profile.samples_count = max(profile.samples_count, len(samples))
        elif layer == DNALayer.MICRO:
            if not context_key:
                raise ValueError("context_key (content_type) is required for MICRO layer.")
            profile.micro_dna[context_key] = avg_fp
        elif layer == DNALayer.CONTEXTUAL:
            if not context_key:
                raise ValueError("context_key is required for CONTEXTUAL layer.")
            profile.contextual_dna[context_key] = avg_fp
        elif layer == DNALayer.TEMPORAL:
            version = DNAVersion(
                version_id=str(uuid.uuid4()),
                layer=DNALayer.TEMPORAL,
                fingerprint=dict(avg_fp),
                label=context_key or f"v{len(profile.versions) + 1}",
            )
            profile.versions.append(version)

        return profile

    def create_version(self, name: str, label: str = "", layer: DNALayer = DNALayer.MACRO) -> DNAVersion:
        """Snapshot the current macro fingerprint as a named version for A/B testing."""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found.")
        profile = self.profiles[name]
        fp = dict(profile.macro_dna or profile.fingerprint)
        version = DNAVersion(
            version_id=str(uuid.uuid4()),
            layer=layer,
            fingerprint=fp,
            label=label or f"v{len(profile.versions) + 1}",
        )
        profile.versions.append(version)
        return version

    def interpolate(self, profile_a: str, profile_b: str, weight_a: float = 0.5,
                    new_name: str = "") -> DNAProfile:
        """Blend two DNA profiles into a hybrid voice.

        Parameters
        ----------
        profile_a, profile_b : Names of existing profiles to blend.
        weight_a             : Weight for profile A (0-1). Profile B gets (1 - weight_a).
        new_name             : Name for the blended profile.
        """
        if profile_a not in self.profiles:
            raise ValueError(f"Profile '{profile_a}' not found.")
        if profile_b not in self.profiles:
            raise ValueError(f"Profile '{profile_b}' not found.")
        if not 0.0 <= weight_a <= 1.0:
            raise ValueError("weight_a must be between 0 and 1.")

        fp_a = self.profiles[profile_a].fingerprint
        fp_b = self.profiles[profile_b].fingerprint
        weight_b = 1.0 - weight_a

        blended: Dict[str, float] = {}
        for dim in DNA_DIMENSIONS:
            blended[dim] = round(fp_a.get(dim, 0) * weight_a + fp_b.get(dim, 0) * weight_b, 4)

        blended_name = new_name or f"{profile_a}+{profile_b}"
        new_profile = DNAProfile(
            name=blended_name,
            fingerprint=blended,
            macro_dna=blended,
            samples_count=self.profiles[profile_a].samples_count + self.profiles[profile_b].samples_count,
        )
        self.profiles[blended_name] = new_profile
        return new_profile

    def detect_drift(self, text: str, profile_name: str,
                     baseline_version_idx: int = 0) -> List[DriftAlert]:
        """Compare text fingerprint to the profile baseline and return drift alerts.

        Parameters
        ----------
        text                 : New content to test.
        profile_name         : Profile to compare against.
        baseline_version_idx : Index into versions list to use as baseline
                               (0 = earliest snapshot). Falls back to macro_dna.
        """
        if profile_name not in self.profiles:
            return []
        profile = self.profiles[profile_name]
        threshold = profile.drift_threshold_pct

        # Determine baseline fingerprint
        if profile.versions and baseline_version_idx < len(profile.versions):
            baseline_fp = profile.versions[baseline_version_idx].fingerprint
        else:
            baseline_fp = profile.macro_dna or profile.fingerprint

        current_fp = self.analyze_sample(text)
        alerts: List[DriftAlert] = []

        for dim in DNA_DIMENSIONS:
            baseline_val = baseline_fp.get(dim, 0)
            current_val = current_fp.get(dim, 0)
            denom = abs(baseline_val) if abs(baseline_val) > _EPSILON else _EPSILON
            delta_pct = abs(current_val - baseline_val) / denom * 100
            alerts.append(DriftAlert(
                profile_name=profile_name,
                dimension=dim,
                baseline=baseline_val,
                current=current_val,
                delta_pct=round(delta_pct, 1),
                threshold_pct=threshold,
                exceeded=delta_pct > threshold,
            ))

        return alerts

    def score(self, text: str, profile_name: str,
              layer: DNALayer = DNALayer.MACRO,
              content_type: str = "") -> Dict:
        """Score how closely a text matches a DNA profile layer. Returns 0-100.

        Parameters
        ----------
        layer        : Which layer fingerprint to score against.
        content_type : Required when layer=MICRO; used as the micro_dna key.
        """
        if profile_name not in self.profiles:
            return {"error": f"Profile '{profile_name}' not found", "score": 0}

        profile = self.profiles[profile_name]

        # Select the target fingerprint based on layer
        if layer == DNALayer.MICRO and content_type and content_type in profile.micro_dna:
            target_fp = profile.micro_dna[content_type]
        elif layer == DNALayer.CONTEXTUAL and content_type and content_type in profile.contextual_dna:
            target_fp = profile.contextual_dna[content_type]
        else:
            target_fp = profile.macro_dna or profile.fingerprint

        if not target_fp:
            return {"error": f"Profile '{profile_name}' has no fingerprint data", "score": 0}

        current = self.analyze_sample(text)

        # Compute per-dimension similarity (1 - normalized distance)
        dim_scores = {}
        for dim in DNA_DIMENSIONS:
            target = target_fp.get(dim, 0)
            actual = current.get(dim, 0)
            max_val = max(abs(target), abs(actual), _MIN_SCORE_DIVISOR)
            similarity = 1.0 - min(abs(target - actual) / max_val, 1.0)
            dim_scores[dim] = round(similarity * 100, 1)

        overall = round(sum(dim_scores.values()) / len(dim_scores), 1)

        return {
            "overall_score": overall,
            "dimension_scores": dim_scores,
            "profile_name": profile_name,
            "layer": layer.value,
            "samples_analyzed": profile.samples_count,
            "current_fingerprint": current,
            "target_fingerprint": target_fp,
        }

    def get_profile_summary(self, profile_name: str) -> Optional[str]:
        """Generate a natural-language description of a DNA profile for agent prompts."""
        if profile_name not in self.profiles:
            return None
        profile = self.profiles[profile_name]
        fp = profile.macro_dna or profile.fingerprint
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
        base = f"Voice profile: {', '.join(parts)}." if parts else "Neutral professional voice."

        extras = []
        if profile.micro_dna:
            extras.append(f"format-specific micro-layers: {', '.join(profile.micro_dna.keys())}")
        if profile.versions:
            extras.append(f"{len(profile.versions)} temporal snapshot(s)")
        if extras:
            base += f" ({'; '.join(extras)})"
        return base


# Singleton
dna_engine = DNAEngine()
