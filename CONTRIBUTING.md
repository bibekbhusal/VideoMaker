# Contributing

Thanks for your interest in improving Nepali Narration Video! To get started:

1. **Set up the environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. **Lint & test before sending changes**
   ```bash
   ruff check src
   python -m compileall src
   ```
3. **Git workflow**
   - Create a topic branch off `master`.
   - Keep commits focused and reference issues where appropriate.
   - Update `CHANGELOG.md` and documentation when behavior changes.

By contributing you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
