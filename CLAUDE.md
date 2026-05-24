# Better Benchmark - Corpus Invariant Validation

## Validating the repro corpus

Before committing changes that touch `repros/`, run:

```bash
python scripts/validate_corpus_invariants.py
```

This enforces hard invariants (exit 1 on failure) and soft invariants (warnings).

## Wiring into CI / pre-commit

### Option A: Git pre-commit hook

```bash
# .git/hooks/pre-commit (or via pre-commit framework)
#!/bin/sh
if git diff --cached --name-only | grep -q '^repros/'; then
    python scripts/validate_corpus_invariants.py || exit 1
fi
```

### Option B: GitHub Actions (recommended for CI)

```yaml
# .github/workflows/corpus-check.yml
name: Corpus invariants
on:
  pull_request:
    paths: ['repros/**']
  push:
    paths: ['repros/**']
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: python scripts/validate_corpus_invariants.py
```

### Option C: Claude Code hook (settings.json)

Add to `.claude/settings.json`:
```json
{
  "hooks": {
    "pre-commit": [
      {
        "match": "repros/",
        "command": "python scripts/validate_corpus_invariants.py"
      }
    ]
  }
}
```
