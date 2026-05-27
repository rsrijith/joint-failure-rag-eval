"""joint-failure-rag-eval: cross-judge joint failure of RAG faithfulness evaluators."""

from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.0.1"

# Auto-load .env from repo root if present, so scripts/CLIs don't need to
# re-export keys every shell. .env is gitignored.
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    load_dotenv(_env, override=False)
