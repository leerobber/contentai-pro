# ContentAI Pro

AI-powered content generation API. Runs on local Ollama (TatorTot) or OpenAI.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env — set your CONTENTAI_API_KEY
python main.py
```

API docs: http://localhost:8001/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/blog | Blog post or outline |
| POST | /api/v1/social | Social media posts |
| POST | /api/v1/email | Email campaigns |
| POST | /api/v1/ad | Ad copy (Google, Meta, Display, YouTube) |
| POST | /api/v1/product-description | Product descriptions |

All endpoints require header: `X-API-Key: your-key`

## Brand Voices
`professional` | `casual` | `bold` | `luxury` | `technical`

## Platforms (social)
`twitter` | `linkedin` | `instagram` | `facebook` | `threads`

## Ad Formats
`google_search` | `meta_feed` | `display` | `youtube`
