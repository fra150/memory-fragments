"""Tests for the HybridRetriever (BM25 + embedding fusion)."""

from memory_fragments.models import Fragment
from memory_fragments.retrieval.retriever import HybridRetriever, _min_max_normalise
from memory_fragments.retrieval.indexer import _tokenize


def _fragment(fid: str, content: str) -> Fragment:
    return Fragment(fragment_id=fid, content=content)


def _populated() -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.add_fragments(
        [
            _fragment(
                "a1",
                "I gatti domestici comunicano attraverso miagolii e movimenti della coda.",
            ),
            _fragment(
                "a2",
                "I cani sono animali domestici fedeli che abbaiano per attirare attenzione.",
            ),
            _fragment(
                "a3",
                "La fotosintesi clorofilliana converte la luce solare in energia chimica.",
            ),
        ]
    )
    return retriever


def test_empty_retriever_returns_empty():
    retriever = HybridRetriever()
    assert retriever.retrieve("gatto", top_k=5) == []


def test_keyword_retrieval_finds_matching_fragment():
    retriever = _populated()
    results = retriever.retrieve_keyword("cani abbaiare", top_k=5)
    assert len(results) == 1
    assert results[0][0].fragment_id == "a2"


def test_hybrid_retrieval_returns_results():
    retriever = _populated()
    results = retriever.retrieve("gatti domestici", top_k=3)
    assert len(results) >= 1
    fids = [f.fragment_id for f, _ in results]
    assert "a1" in fids


def test_semantic_retrieval_returns_sorted():
    retriever = _populated()
    results = retriever.retrieve_semantic("fotosintesi energia solare", top_k=3)
    assert len(results) == 3
    # Sorted descending by score
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results():
    retriever = _populated()
    assert len(retriever.retrieve_semantic("luce solare", top_k=1)) == 1


def test_remove_fragment():
    retriever = _populated()
    retriever.remove_fragment("a1")
    fids = [f.fragment_id for f, _ in retriever.retrieve_keyword("gatti", top_k=5)]
    assert "a1" not in fids


def test_rebuild():
    retriever = _populated()
    retriever.rebuild([_fragment("b1", "Contenuto completamente nuovo e diverso.")])
    assert retriever.bm25.fragment_count == 1
    assert retriever.embedding.fragment_count == 1


def test_min_max_normalise():
    from memory_fragments.models import Fragment

    frags = [Fragment(fragment_id=f"n{i}", content=f"content {i}") for i in range(2)]
    scored = [(frags[0], 1.0), (frags[1], 5.0)]
    normalised = _min_max_normalise(scored)
    assert normalised[0][1] == 0.0
    assert normalised[1][1] == 1.0


def test_min_max_normalise_flat_scores():
    from memory_fragments.models import Fragment

    frags = [Fragment(fragment_id="n0", content="content x")]
    normalised = _min_max_normalise([(frags[0], 0.5)])
    assert normalised[0][1] == 1.0


def test_tokenizer():
    assert _tokenize("Hello, World! Don't stop.") == ["hello", "world", "don't", "stop"]


def test_config_defaults():
    retriever = HybridRetriever()
    assert retriever.config.top_k == 5
    assert retriever.config.bm25_weight == 0.4
    assert retriever.config.embedding_weight == 0.6
