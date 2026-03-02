# ContentAI Pro

Multi-agent AI content generation platform with voice DNA fingerprinting, adversarial quality debate, 8-platform content atomization, and real-time trend radar.

## Architecture

```
Research → Writer → Editor → SEO → DNA Score → Debate → Atomizer
                                                           ↓
                                    Twitter · LinkedIn · Instagram · Email
                                    Reddit · YouTube · TikTok · Podcast
```

## Quick Start

```bash
git clone https://github.com/leerobber/leerobber.git
cd leerobber
pip install -r requirements.txt
cp .env.example .env   # add your API key
python run.py           # → http://localhost:8000
```

### Docker

```bash
docker build -t contentai-pro .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-xxx contentai-pro
```

## Features

**Content DNA Engine** — 14-dimension voice fingerprint that learns your writing style from samples. Agents match your voice. New content is scored for consistency.

**Adversarial Agent Debate** — Advocate defends the content, Critic attacks it, Judge rules. Auto-revises and re-debates until it passes.

**Content Atomizer** — One piece → 8 platform-native variants (Twitter thread, LinkedIn, Instagram, Email, Reddit, YouTube script, TikTok, Podcast notes).

**Trend Radar** — Live trending topics from HackerNews, Reddit, Dev.to with niche filtering.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/content/generate` | Full 7-stage pipeline |
| POST | `/api/content/generate/quick` | Quick single-pass generation |
| POST | `/api/content/generate/stream` | SSE streaming pipeline |
| POST | `/api/content/atomize` | Atomize into platform variants |
| POST | `/api/content/debate` | Adversarial quality debate |
| POST | `/api/content/dna/calibrate` | Build voice DNA profile |
| POST | `/api/content/dna/score` | Score content against profile |
| GET | `/api/content/trends` | Trending topics |
| GET | `/api/content/{id}` | Retrieve content by ID |
| GET | `/api/health` | Health check |

## Project Structure

```
run.py                              # Launch script
contentai_pro/
├── main.py                         # FastAPI app factory
├── core/
│   ├── config.py                   # Settings (.env support)
│   ├── events.py                   # Event bus + SSE streaming
│   ├── middleware.py               # Request ID + logging
│   └── database.py                 # Async SQLite storage
├── ai/
│   ├── llm_adapter.py             # Anthropic/OpenAI/Mock adapter
│   ├── orchestrator.py            # 7-stage pipeline orchestrator
│   ├── agents/
│   │   ├── base.py                # Agent interface
│   │   ├── specialists.py         # Research/Writer/Editor/SEO agents
│   │   └── debate.py              # Advocate-Critic-Judge debate
│   ├── dna/engine.py              # Voice fingerprinting (14 dimensions)
│   ├── atomizer/engine.py         # Platform variant generator (8 platforms)
│   └── trends/radar.py            # HN/Reddit/Dev.to scanner
├── modules/
│   └── content/router.py          # API endpoints
static/
├── index.html                      # Dashboard UI
├── css/app.css                     # Steampunk theme
└── js/app.js                       # Frontend controller
```

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

- `mock` — UI testing without API keys (default)
- `anthropic` — Claude via Anthropic API
- `openai` — GPT via OpenAI API

## DNA Dimensions

The Content DNA Engine fingerprints writing across 14 dimensions:

1. Sentence length average
2. Sentence length variance
3. Vocabulary tier (advanced word ratio)
4. Passive voice ratio
5. Question frequency
6. Metaphor density
7. Technical depth
8. Paragraph rhythm
9. Transition word density
10. Contraction ratio
11. First-person usage
12. Exclamation energy
13. List structure ratio
14. Opening hook style

## License

MIT
