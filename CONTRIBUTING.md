# Contributing to modelrack

Thanks for your interest in improving modelrack! 🎛️

## Development setup

```bash
git clone https://github.com/m-555/ModelRack
cd modelrack
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

Run the full local check suite (mirrors CI):

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Ground rules

- **The hub stays ML-free.** Never add `torch`, `diffusers`, `transformers`, `vllm`, or any
  heavy ML dependency to `modelrack`'s runtime deps. Those belong only in server *templates*
  and per-model `requirements.txt` files.
- **Keep server templates self-contained.** They run inside isolated venvs and must not import
  `modelrack`. Only customize `load_model()` / `run_inference()`; keep the FastAPI scaffolding
  identical across templates.
- **Add tests** for new behavior. Prefer the existing fixtures in `tests/conftest.py`.
- **New model types** should follow [docs/adding-a-type.md](docs/adding-a-type.md).

## Commit & PR style

- Small, focused commits with clear messages.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Describe user-facing changes and how you tested them in the PR body.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
