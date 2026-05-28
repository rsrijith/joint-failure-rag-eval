"""joint-failure-rag-eval: cross-judge joint failure of RAG faithfulness evaluators."""

from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.0.1"

# Auto-load .env from repo root if present, so scripts/CLIs don't need to
# re-export keys every shell. .env is gitignored.
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    load_dotenv(_env, override=False)

# Patch mistralai to expose `Mistral` at top level. mistralai 2.x ships
# the class at `mistralai.client.sdk.Mistral` but some downstream packages
# (instructor, langchain-mistralai) import `from mistralai import Mistral`
# which fails. Apply the patch eagerly so any later import works.
try:
    import mistralai as _mistralai
    if not hasattr(_mistralai, "Mistral"):
        from mistralai.client.sdk import Mistral as _Mistral
        _mistralai.Mistral = _Mistral
except Exception:
    pass

# Stub `langchain_community.chat_models.vertexai` (RAGAS imports it eagerly,
# langchain-community 0.4.x removed it in the sunset migration). We never
# actually use VertexAI; the stub just needs to satisfy the import.
try:
    import sys
    import types
    _stub_name = "langchain_community.chat_models.vertexai"
    if _stub_name not in sys.modules:
        _stub = types.ModuleType(_stub_name)
        class _StubChatVertexAI:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("VertexAI is stubbed in this venv")
        _stub.ChatVertexAI = _StubChatVertexAI
        sys.modules[_stub_name] = _stub
except Exception:
    pass
