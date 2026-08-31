"""Tests for the StaticArchive (immutable fragment store)."""

import pytest

from memory_fragments.archive import StaticArchive
from memory_fragments.models import Fragment, FragmentConditions, FragmentMetadata, FragmentStatus


def make_fragment(fragment_id: str, content: str, quality: float = 0.8, tags=None, topic: str = "test") -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(topic=topic, quality=quality, tags=tags or []),
    )


class TestAdd:
    def test_add_and_get_roundtrip(self):
        archive = StaticArchive()
        frag = make_fragment("frag-1", "Some knowledge content")
        archive.add(frag)

        stored = archive.get("frag-1")
        assert stored is not None
        assert stored.fragment_id == "frag-1"
        assert stored.content == "Some knowledge content"
        assert stored.checksum  # checksum auto-computed

    def test_get_returns_deep_copy(self):
        archive = StaticArchive()
        frag = make_fragment("frag-2", "Immutable content")
        archive.add(frag)

        stored = archive.get("frag-2")
        stored.content = "MUTATED"

        # Canonical store must be untouched
        assert archive.get("frag-2").content == "Immutable content"

    def test_duplicate_rejected(self):
        archive = StaticArchive()
        archive.add(make_fragment("dup", "first"))
        with pytest.raises(ValueError):
            archive.add(make_fragment("dup", "second"))

    def test_missing_checksum_rejected(self):
        archive = StaticArchive()
        frag = Fragment(fragment_id="no-checksum", content="x")
        # __post_init__ auto-computes the checksum; simulate a corrupted
        # fragment by clearing it before adding to the archive.
        frag.checksum = ""
        with pytest.raises(ValueError):
            archive.add(frag)

    def test_oversized_fragment_rejected(self):
        archive = StaticArchive()
        long_content = "x" * 10_001
        with pytest.raises(ValueError):
            archive.add(make_fragment("long", long_content))

    def test_quality_clamped(self):
        frag = make_fragment("q", "content", quality=1.7)
        assert frag.metadata.quality == 1.0

        frag2 = make_fragment("q2", "content", quality=-0.4)
        assert frag2.metadata.quality == 0.0


class TestDeleteAndList:
    def test_delete_soft_archives(self):
        archive = StaticArchive()
        archive.add(make_fragment("del-1", "content"))
        assert archive.delete("del-1") is True
        assert archive.get("del-1").status == FragmentStatus.ARCHIVED

    def test_delete_missing_returns_false(self):
        archive = StaticArchive()
        assert archive.delete("nope") is False

    def test_list_filters_by_topic(self):
        archive = StaticArchive()
        archive.add(make_fragment("a", "one", topic="physics"))
        archive.add(make_fragment("b", "two", topic="biology"))
        topics = [f.metadata.topic for f in archive.list(topic="physics")]
        assert topics == ["physics"]

    def test_list_filters_by_status(self):
        archive = StaticArchive()
        archive.add(make_fragment("a", "one"))
        archive.delete("a")
        archived = archive.list(status=FragmentStatus.ARCHIVED)
        active = archive.list(status=FragmentStatus.ACTIVE)
        assert len(archived) == 1
        assert len(active) == 0

    def test_list_filters_by_tags(self):
        archive = StaticArchive()
        archive.add(make_fragment("a", "one", tags=["energy", "physics"]))
        archive.add(make_fragment("b", "two", tags=["biology"]))
        matched = archive.list(tags=["Physics"])  # case-insensitive tag match
        assert [f.fragment_id for f in matched] == ["a"]


class TestSearch:
    def test_search_by_keywords(self):
        archive = StaticArchive()
        archive.add(make_fragment("k1", "Photosynthesis uses sunlight energy."))
        archive.add(make_fragment("k2", "Quantum entanglement is strange."))
        results = archive.search_by_keywords(["photosynthesis"])
        assert [f.fragment_id for f in results] == ["k1"]

    def test_search_by_keywords_empty(self):
        archive = StaticArchive()
        assert archive.search_by_keywords([]) == []

    def test_search_by_quality(self):
        archive = StaticArchive()
        archive.add(make_fragment("q-low", "low", quality=0.3))
        archive.add(make_fragment("q-high", "high", quality=0.9))
        results = archive.search_by_quality(0.8)
        assert [f.fragment_id for f in results] == ["q-high"]

    def test_count_and_contains(self):
        archive = StaticArchive()
        archive.add(make_fragment("c1", "one"))
        archive.add(make_fragment("c2", "two"))
        assert archive.count() == 2
        assert len(archive) == 2
        assert "c1" in archive
        assert "c3" not in archive
        assert archive.all_fragment_ids() == ["c1", "c2"]


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        archive = StaticArchive()
        archive.add(make_fragment("s1", "content one", quality=0.7, tags=["a"]))
        archive.add(make_fragment("s2", "content two", quality=0.9, tags=["b"]))

        restored = StaticArchive.from_dict(archive.to_dict())

        assert restored.count() == 2
        assert restored.get("s1").content == "content one"
        assert restored.get("s1").metadata.quality == 0.7
        assert restored.get("s1").checksum == archive.get("s1").checksum
