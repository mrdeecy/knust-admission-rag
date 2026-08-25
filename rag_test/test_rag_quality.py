"""RAG quality regression suite: golden-set questions -> RAGAS assertions.

Every test in this file calls a live LLM and the live Pinecone indexes, so
they're all marked `llm` and excluded from the default `pytest` run (see
`pytest.ini`). Run them explicitly with `pytest tests -m llm`.

`AdvancedRetriever`'s multi-query step samples at temperature=0.7, so which
chunks get retrieved (and therefore what the judge considers "supported")
can shift slightly run to run. We call `retrieve(..., use_multi_query=False)`
here and use thresholds with margin (`>= 0.5`, not `== 1.0`) so the suite
stays a meaningful regression signal instead of a coin flip.
"""
import pytest
import rag_test.retrieval_pipeline as rag
from conftest import load_golden_set
from ragas import SingleTurnSample

THRESHOLDS = {
    "context_recall": 0.5,
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
}


def _retrieve_no_multi_query(query: str):
    return rag.retriever.retrieve(query, use_multi_query=False)


@pytest.mark.llm
@pytest.mark.parametrize(
    "item", load_golden_set(), ids=lambda item: item["question"][:60]
)
def test_answer_is_grounded_and_relevant(item, ragas_metrics):
    answer, contexts = rag.generate_answer(
        item["question"], retriever_fn=_retrieve_no_multi_query
    )

    sample = SingleTurnSample(
        user_input=item["question"],
        retrieved_contexts=contexts,
        response=answer,
        reference=item["ground_truth"],
    )

    scores = {
        name: ragas_metrics[name].single_turn_score(sample)
        for name in THRESHOLDS
    }

    failures = [
        f"{name}={score:.2f} < {THRESHOLDS[name]}"
        for name, score in scores.items()
        if score < THRESHOLDS[name]
    ]
    assert not failures, (
        f"question: {item['question']!r}\n"
        f"answer: {answer}\n"
        f"failed metrics: {', '.join(failures)}\n"
        f"all scores: {scores}"
    )
