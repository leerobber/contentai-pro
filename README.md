# contentai-pro

> *Multi-agent content engine — adversarial debate before every publish, DNA fingerprinting, 8 platforms simultaneously.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Sovereign Core](https://img.shields.io/badge/Sovereign_Core-priority_0-00ff88?style=flat-square)](https://github.com/leerobber/sovereign-core)

---

## What This Is

contentai-pro is a **multi-agent AI content engine** that runs adversarial debate between AI models before publishing anything, generates content for 8 platforms simultaneously, fingerprints every piece of content with a unique DNA signature, and monitors trends to generate content automatically.

All generation runs on local hardware through the **Sovereign Core gateway** — no OpenAI API bills, no rate limits, no external dependency.

---

## How It Works

```
┌──────────────────────────────────────────────────────┐
│                CONTENT PIPELINE                      │
│                                                      │
│  1. Trend Radar detects signal                       │
│  2. Agent A generates content draft                  │
│  3. Agent B argues against it (adversarial debate)   │
│  4. Agent A revises based on critique                │
│  5. DNA Fingerprinter tags content uniquely          │
│  6. 8-Platform Atomizer adapts for each format       │
│  7. Output: Twitter/X, LinkedIn, Instagram,          │
│             TikTok, YouTube, Blog, Newsletter, More  │
│                                                      │
│  No publish without debate. Quality by design.       │
└──────────────────────────────────────────────────────┘
```

**Adversarial debate:** Two models argue before anything ships. Model A proposes. Model B critiques. Model A revises. This catches weak content before it's published — not after.

---

## Core Features

| Feature | What It Does |
|---------|-------------|
| **DNA Fingerprinting** | Every piece of content gets a unique signature — track origination, detect plagiarism, prove authenticity |
| **Adversarial Debate** | Two AI models debate the content before publish — quality by pressure, not by hoping |
| **8-Platform Atomizer** | One input → 8 platform-native outputs automatically adapted for format, tone, and length |
| **Trend Radar** | Monitors signals and generates relevant content proactively |
| **Sovereign Provider** | All inference through sovereign-core gateway — local GPU, no cloud |

---

## Sovereign Core Integration

```python
# contentai uses sovereign-core as priority-0 provider
# .env
SOVEREIGN_GATEWAY_URL=http://localhost:8000

# Falls back to other providers only if gateway is unreachable
# Routing: RTX 5050 → Radeon 780M → Ryzen CPU
```

---

## Quickstart

```bash
git clone https://github.com/leerobber/contentai-pro
cd contentai-pro
docker-compose up -d
cp .env.example .env
# Set SOVEREIGN_GATEWAY_URL
python -m uvicorn main:app --reload
```

---

## Part of the Sovereign Stack

| Repo | Role |
|------|------|
| [sovereign-core](https://github.com/leerobber/sovereign-core) | Gateway + KAIROS engine |
| [DGM](https://github.com/leerobber/DGM) | Darwin Gödel Machine |
| [HyperAgents](https://github.com/leerobber/HyperAgents) | Self-referential swarm agents |
| [Honcho](https://github.com/leerobber/Honcho) | Mission control dashboard |
| **contentai-pro** | Multi-agent content engine — sovereign provider priority |

---

## Built By

**Terry Lee** — Douglasville, GA  
Self-taught systems architect. No team. No institution. Just architecture.

*Self-taught. Self-funded. Self-improving — just like the systems I build.*
