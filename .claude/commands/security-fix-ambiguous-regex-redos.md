---
name: security-fix-ambiguous-regex-redos
description: Workflow command scaffold for security-fix-ambiguous-regex-redos in contentai-pro.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /security-fix-ambiguous-regex-redos

Use this workflow when working on **security-fix-ambiguous-regex-redos** in `contentai-pro`.

## Goal

Fixes ambiguous or vulnerable regular expressions in the DNA engine to prevent ReDoS (Regular Expression Denial of Service) vulnerabilities.

## Common Files

- `contentai_pro/ai/dna/engine.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Identify the vulnerable or ambiguous regex in contentai_pro/ai/dna/engine.py.
- Update the regex to a safer, unambiguous pattern.
- Commit the fix with a message referencing ReDoS or regex ambiguity.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.