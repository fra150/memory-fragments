"""Test del retrieval — BM25, EmbeddingIndexer (fallback) e HybridRetriever."""

from memory_fragments.models import Fragment, FragmentMetadata
from memory_fragments.retrieval.indexer import BM25Indexer, EmbeddingIndexer
from memory_fragments.retrieval.retriever import HybridRetriever


def _fragment(fragment_id: str, content: str, quality: float = 0.8) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(quality=quality),
    )


FRAGMENTS = [
    _fragment("F-1", "il gatto dorme sul divano"),
    _fragment("F-2", "il cane abbaia nel cortile"),
    _fragment("F-3", "il gatto caccia i topi di notte"),
    _fragment("F-4", "la pianta cresce in primavera"),
]


class TestBM25Indexer:
    def test_search_returns_most_relevant(self):
        indexer = BM25Indexer()
        indexer.index_fragments(FRAGMENTS)
        results = indexer.search("gatto", top_k=5)
        assert results
        assert results[0][0].fragment_id in {"F-1", "F-3"}

    def test_incremental_add_remove(self):
        indexer = BM25Indexer()
        indexer.add_fragment(FRAGMENTS[0])
        assert indexer.fragment_count == 1
        indexer.remove_fragment("F-1")
        assert indexer.fragment_count == 0
        assert indexer.search("gatto") == []

    def test_empty_index(self):
        assert BM25Indexer().search("gatto") == []


class TestEmbeddingIndexer:
    def test_fallback_deterministic(self):
        a = EmbeddingIndexer()
        b = EmbeddingIndexer()
        a.index_fragments(FRAGMENTS)
        b.index_fragments(FRAGMENTS)
        ra = a.search("gatto dorme")
        rb = b.search("gatto dorme")
        assert [f.fragment_id for f, _ in ra] == [f.fragment_id for f, _ in rb]

    def test_search_returns_something(self):
        indexer = EmbeddingIndexer()
        indexer.index_fragments(FRAGMENTS)
        results = indexer.search("gatto", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(s, float) for _, s in results)

    def test_remove(self):
        indexer = EmbeddingIndexer()
        indexer.add_fragment(FRAGMENTS[0])
        indexer.remove_fragment("F-1")
        assert indexer.fragment_count == 0


class TestHybridRetriever:
    def test_retrieve_sorted(self):
        retriever = HybridRetriever()
        retriever.add_fragments(FRAGMENTS)
        results = retriever.retrieve("gatto", top_k=3)
        assert len(results) <= 3
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_keyword_only(self):
        retriever = HybridRetriever()
        retriever.add_fragments(FRAGMENTS)
        results = retriever.retrieve_keyword("cane", top_k=5)
        assert results and results[0][0].fragment_id == "F-2"

    def test_retrieve_semantic_only(self):
        retriever = HybridRetriever()
        retriever.add_fragments(FRAGMENTS)
        results = retriever.retrieve_semantic("gatto", top_k=2)
        assert 1 <= len(results) <= 2

    def test_remove_and_rebuild(self):
        retriever = HybridRetriever()
        retriever.add_fragments(FRAGMENTS)
        retriever.remove_fragment("F-1")
        assert "F-1" not in [f.fragment_id for f, _ in retriever.retrieve("gatto", top_k=10)]

        retriever.rebuild(FRAGMENTS)
        results = retriever.retrieve("gatto", top_k=10)
        assert {f.fragment_id for f, _ in results} >= {"F-1"}

    def test_retrieve_with_conflicts_no_archive(self):
        retriever = HybridRetriever()
        retriever.add_fragments(FRAGMENTS)
        results, reports = retriever.retrieve_with_conflicts("gatto")
        assert results
        assert reports == []

    def test_retrieve_empty(self):
        retriever = HybridRetriever()
        assert retriever.retrieve("gatto") == []
