---
name: code-cleanup-remove-unused-imports
description: Workflow command scaffold for code-cleanup-remove-unused-imports in contentai-pro.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /code-cleanup-remove-unused-imports

Use this workflow when working on **code-cleanup-remove-unused-imports** in `contentai-pro`.

## Goal

Removes unused imports across multiple Python modules to maintain code cleanliness and reduce technical debt.

## Common Files

- `contentai_pro/ai/agents/debate.py`
- `contentai_pro/ai/dna/engine.py`
- `contentai_pro/ai/llm_adapter.py`
- `contentai_pro/ai/orchestrator.py`
- `contentai_pro/ai/trends/radar.py`
- `contentai_pro/core/cache.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Identify unused imports in multiple Python files.
- Remove the unused imports from each file.
- Commit changes with a message referencing code cleanup.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.