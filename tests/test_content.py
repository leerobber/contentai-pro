"""Basic tests -- mock LLM calls."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch

def test_blog_structure():
    with patch("core.content_engine.generate", return_value="mock blog content"):
        from generators.blog import generate_blog
        result = generate_blog("AI trends 2025")
        assert result["type"] == "blog_post"
        assert result["topic"] == "AI trends 2025"
        assert result["content"] == "mock blog content"
        print("PASS: test_blog_structure")

def test_social_structure():
    with patch("core.content_engine.generate", return_value="mock post"):
        from generators.social import generate_social
        result = generate_social("product launch", platform="twitter")
        assert result["type"] == "social_post"
        assert result["platform"] == "twitter"
        assert len(result["posts"]) == 1
        print("PASS: test_social_structure")

def test_email_structure():
    with patch("core.content_engine.generate", return_value="mock email"):
        from generators.email import generate_email
        result = generate_email("Black Friday sale", cta="Shop Now")
        assert result["type"] == "email"
        assert result["cta"] == "Shop Now"
        print("PASS: test_email_structure")

if __name__ == "__main__":
    test_blog_structure()
    test_social_structure()
    test_email_structure()
    print("All tests passed.")
