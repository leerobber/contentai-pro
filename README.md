# contentai-pro

> Multi-Agent AI Content Engine — DNA fingerprinting, adversarial debate, 8-platform atomizer, trend radar.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)

---

## Architecture

9-stage pipeline: Research → Write → Fact-Check → Edit → SEO → Headline → DNA Score → Debate → Atomize

### LLM Provider Priority
1. **Sovereign Core Gateway** — local GPU cluster (RTX 5050 → Radeon 780M → Ryzen 7) — **free, zero latency**
2. Anthropic Claude — cloud fallback
3. OpenAI GPT-4o — final fallback
4. Mock — testing/dev

### Components

| Module | Description |
|--------|-------------|
| `ai/orchestrator.py` | 9-stage pipeline with per-stage error handling |
| `ai/agents/debate.py` | Advocate + Critic run in parallel; Judge scores |
| `ai/dna/engine.py` | 14-dimension voice fingerprinting |
| `ai/atomizer/engine.py` | 8-platform content atomizer |
| `ai/trends/radar.py` | Trend radar |
| `ai/llm_sovereign.py` | Sovereign Core gateway adapter |
| `ai/llm_adapter.py` | Unified LLM interface with Sovereign priority |

---

## Quick Start

```bash
git clone https://github.com/leerobber/contentai-pro
cd contentai-pro
pip install -r requirements.txt

# Optional: point at your Sovereign Core gateway
export SOVEREIGN_GATEWAY_URL=http://localhost:8000
export SOVEREIGN_ENABLED=true

# Run
python run.py
```

## Environment Variables

```bash
# Sovereign Core (priority 0 — free local GPU)
SOVEREIGN_GATEWAY_URL=http://localhost:8000
SOVEREIGN_ENABLED=true

# Cloud fallbacks
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# App settings
APP_ENV=development
LOG_LEVEL=INFO
```

## Integration with Sovereign Core

contentai-pro automatically routes all LLM calls through your local GPU cluster when `SOVEREIGN_ENABLED=true`. Set `SOVEREIGN_GATEWAY_URL` to your gateway address.

The `llm_sovereign.py` adapter handles:
- Automatic health checking with cache TTL
- Graceful fallback to cloud on gateway unavailability
- Token counting and cost tracking (sovereign = $0.00)
- Connection pooling via `httpx.AsyncClient`

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --tb=short
```
