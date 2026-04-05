```markdown
# contentai-pro Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core coding conventions, workflows, and automation patterns used in the `contentai-pro` Python codebase. You'll learn how to maintain code cleanliness, address security vulnerabilities, and follow the project's established style for file organization, imports, and exports. This guide also covers how to trigger and execute common maintenance and security workflows using suggested commands.

## Coding Conventions

### File Naming
- **Style:** `snake_case`
- **Example:**  
  ```
  contentai_pro/ai/llm_adapter.py
  contentai_pro/core/cache.py
  ```

### Import Style
- **Relative imports** are preferred within the package.
- **Example:**
  ```python
  from .utils import parse_config
  from ..core.cache import CacheManager
  ```

### Export Style
- **Named exports** are used; avoid wildcard (`*`) exports.
- **Example:**
  ```python
  __all__ = ['DebateAgent', 'Orchestrator']
  ```

### Commit Messages
- **Freeform** style, no strict prefixes.
- **Average length:** ~49 characters.
- **Example:**  
  ```
  Remove unused imports from core modules
  Fix ambiguous regex in DNA engine to prevent ReDoS
  ```

## Workflows

### Code Cleanup: Remove Unused Imports
**Trigger:** When you want to clean up the codebase by removing unused imports.  
**Command:** `/cleanup-unused-imports`

1. Identify unused imports in multiple Python files.
2. Remove the unused imports from each file.
3. Commit changes with a message referencing code cleanup.

**Files commonly involved:**
- `contentai_pro/ai/agents/debate.py`
- `contentai_pro/ai/dna/engine.py`
- `contentai_pro/ai/llm_adapter.py`
- `contentai_pro/ai/orchestrator.py`
- `contentai_pro/ai/trends/radar.py`
- `contentai_pro/core/cache.py`
- `contentai_pro/core/database.py`
- `contentai_pro/core/metrics.py`
- `contentai_pro/modules/content/router.py`

**Example:**
```python
# Before cleanup
import os
import sys
from .utils import parse_config

def run():
    print("Running agent")
```
```python
# After cleanup (removed unused imports)
from .utils import parse_config

def run():
    print("Running agent")
```

### Security Fix: Ambiguous Regex (ReDoS)
**Trigger:** When you discover a ReDoS or ambiguous regex vulnerability in the DNA engine.  
**Command:** `/fix-regex-redos`

1. Identify the vulnerable or ambiguous regex in `contentai_pro/ai/dna/engine.py`.
2. Update the regex to a safer, unambiguous pattern.
3. Commit the fix with a message referencing ReDoS or regex ambiguity.

**Example:**
```python
# Before fix (ambiguous regex)
import re
pattern = re.compile(r'(a+)+b')

# After fix (safe regex)
import re
pattern = re.compile(r'a+b')
```

## Testing Patterns

- **Framework:** Unknown (not detected)
- **Test file pattern:** `*.test.ts` (suggests some TypeScript testing, possibly for a frontend or API layer)
- **Python testing conventions:** Not explicitly detected; recommend following standard Python testing practices (e.g., `pytest` or `unittest`) for any new tests.

## Commands

| Command                | Purpose                                              |
|------------------------|------------------------------------------------------|
| /cleanup-unused-imports| Remove unused imports from Python modules            |
| /fix-regex-redos       | Fix ambiguous or vulnerable regex patterns (ReDoS)   |
```
