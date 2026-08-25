"""Shared pytest fixtures for the RAG quality suite.

`retrieval_pipeline.py` lives at the repo root, one level above this `tests/`
directory. `python -m pytest tests` (run from the repo root, as the README
and Part 4 notebook do) already puts the repo root on `sys.path`, but we
insert it explicitly too so the suite also works when pytest is invoked from
elsewhere.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rag_test.retrieval_pipeline as rag  # noqa: E402

from langchain_openai import OpenAIEmbeddings  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics._answer_relevance import ResponseRelevancy  # noqa: E402
from ragas.metrics._context_precision import LLMContextPrecisionWithReference  # noqa: E402
from ragas.metrics._context_recall import LLMContextRecall  # noqa: E402
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: calls a live LLM/vector DB (slow, costs money; run explicitly with -m llm)"
    )


def load_golden_set():
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_set():
    return load_golden_set()


@pytest.fixture(scope="session")
def retriever():
    return rag.retriever


@pytest.fixture(scope="session")
def ragas_metrics():
    """Build each RAGAS metric once per test session — the judge LLM and
    embedding model have real setup cost we don't want to pay per test."""
    evaluator_llm = llm_factory(rag.CHAT_MODEL, client=rag.openai_client, temperature=0)
    evaluator_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=rag.EMBEDDING_MODEL, api_key=rag.OPENAI_API_KEY)
    )
    return {
        "context_precision": LLMContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": LLMContextRecall(llm=evaluator_llm),
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
    }
