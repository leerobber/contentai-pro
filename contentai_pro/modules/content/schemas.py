"""Enhanced Pydantic schemas with input validation for the content API."""
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ContentType(str, Enum):
    blog_post = "blog_post"
    article = "article"
    whitepaper = "whitepaper"
    case_study = "case_study"
    newsletter = "newsletter"
    social_post = "social_post"
    press_release = "press_release"
    landing_page = "landing_page"


class ToneType(str, Enum):
    professional = "professional"
    casual = "casual"
    formal = "formal"
    friendly = "friendly"
    authoritative = "authoritative"
    conversational = "conversational"


class Platform(str, Enum):
    twitter = "twitter"
    linkedin = "linkedin"
    instagram = "instagram"
    email = "email"
    reddit = "reddit"
    youtube = "youtube"
    tiktok = "tiktok"
    podcast = "podcast"


# Minimum word counts per content type
_CONTENT_TYPE_MIN_WORDS = {
    ContentType.blog_post: 300,
    ContentType.article: 400,
    ContentType.whitepaper: 500,
    ContentType.case_study: 400,
    ContentType.newsletter: 200,
    ContentType.social_post: 100,
    ContentType.press_release: 200,
    ContentType.landing_page: 200,
}


class GenerateRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The content topic (3-500 characters, must contain at least 2 words).",
        examples=["How AI is transforming enterprise content creation"],
    )
    content_type: ContentType = Field(
        default=ContentType.blog_post,
        description="Type of content to generate.",
    )
    audience: str = Field(
        default="tech professionals",
        max_length=200,
        description="Target audience description.",
    )
    tone: ToneType = Field(
        default=ToneType.professional,
        description="Desired tone of voice.",
    )
    word_count: int = Field(
        default=1200,
        ge=100,
        le=10000,
        description="Target word count (100-10000).",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="SEO keywords to include (max 20).",
    )
    dna_profile: Optional[str] = Field(
        default=None,
        description="Name of a registered DNA voice profile to match.",
    )
    enable_debate: bool = Field(default=True, description="Run adversarial debate stage.")
    enable_atomizer: bool = Field(default=True, description="Run platform atomizer stage.")
    atomizer_platforms: Optional[List[Platform]] = Field(
        default=None,
        description="Platforms to generate variants for. Defaults to all 8.",
    )
    skip_stages: List[str] = Field(
        default_factory=list,
        description="Pipeline stage names to skip.",
    )
    fail_policy: str = Field(
        default="skip",
        description="Error handling policy: 'skip' continues on stage failure, 'fail_fast' aborts.",
    )

    @field_validator("topic")
    @classmethod
    def topic_must_have_two_words(cls, v: str) -> str:
        if len(v.split()) < 2:
            raise ValueError("Topic must contain at least 2 words.")
        return v

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, v: List[str]) -> List[str]:
        if len(v) > 20:
            raise ValueError("Maximum 20 keywords allowed.")
        return [kw.lower().strip() for kw in v if kw.strip()]

    @model_validator(mode="after")
    def check_word_count_for_content_type(self) -> "GenerateRequest":
        min_words = _CONTENT_TYPE_MIN_WORDS.get(self.content_type, 100)
        if self.word_count < min_words:
            raise ValueError(
                f"Content type '{self.content_type}' requires at least {min_words} words, "
                f"got {self.word_count}."
            )
        return self


class DNACalibrateRequest(BaseModel):
    name: str = Field(
        ...,
        description="Profile name (alphanumeric, underscores, and hyphens only).",
        examples=["brand_voice_2025"],
    )
    samples: List[str] = Field(
        ...,
        min_length=3,
        description="Writing samples for calibration (minimum 3, each at least 100 characters).",
    )

    @field_validator("name")
    @classmethod
    def validate_profile_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Profile name must be alphanumeric with underscores or hyphens only.")
        return v

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v: List[str]) -> List[str]:
        if len(v) < 3:
            raise ValueError("Minimum 3 writing samples required.")
        for i, sample in enumerate(v):
            if len(sample) < 100:
                raise ValueError(f"Sample {i + 1} must be at least 100 characters long.")
        return v
