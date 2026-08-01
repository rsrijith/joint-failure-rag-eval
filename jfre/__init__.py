"""jfre: audit a RAG faithfulness judge for the citation-attribution gap.

The fast path is dependency-light. ``audit_judge`` and the attribution-aware
``fix`` use only the standard library, so you can audit your own judge without
installing the reference judges (which pull torch / transformers / API SDKs).

    from jfre import make_seed, relocate_citations, audit_judge
    from jfre.fix import ATTRIBUTION_AWARE_PROMPT, make_attribution_judge

Install the reference judges (Claude, Mistral, HHEM, MiniCheck, AlignScore,
RAGAS-style, FaithJudge-style) only if you want them: ``pip install "fidecite[judges]"``.

Everything below the public API is optional-import convenience for the research
scripts in ``scripts/``. Each block is guarded, so importing ``jfre`` in an
environment that has none of the heavy dependencies installed must never fail.
That guarantee is what the clean-environment install test protects, and it is
the exact bug that shipped in 0.0.1: a module-level ``from dotenv import
load_dotenv`` made ``import jfre`` fail for anyone who installed the package.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Optional: load a local .env if both the file and python-dotenv are present.
# Never required for the BYO-judge path.
try:  # pragma: no cover - convenience only
    from pathlib import Path

    from dotenv import load_dotenv

    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=False)
except Exception:  # pragma: no cover
    pass

# Patch mistralai to expose `Mistral` at top level. mistralai 2.x ships the class
# at `mistralai.client.sdk.Mistral` but some downstream packages (instructor,
# langchain-mistralai) import `from mistralai import Mistral`, which fails. Apply
# the patch eagerly so any later import works. Needed by the Mistral reference
# judge; a no-op when the SDK is not installed.
try:  # pragma: no cover - reference-judge environments only
    import mistralai as _mistralai

    if not hasattr(_mistralai, "Mistral"):
        from mistralai.client.sdk import Mistral as _Mistral

        _mistralai.Mistral = _Mistral
except Exception:  # pragma: no cover
    pass

# Stub `langchain_community.chat_models.vertexai` (RAGAS imports it eagerly;
# langchain-community 0.4.x removed it in the sunset migration). VertexAI is
# never used here, so the stub only needs to satisfy the import. Needed by the
# RAGAS-style reference judge; a no-op when langchain is not installed.
try:  # pragma: no cover - reference-judge environments only
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
except Exception:  # pragma: no cover
    pass

from jfre.audit import AuditResult, Judge, audit_judge, make_seed, relocate_citations

__all__ = [
    "AuditResult",
    "Judge",
    "audit_judge",
    "make_seed",
    "relocate_citations",
    "__version__",
]
