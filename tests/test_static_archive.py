"""Test della StaticArchive — store immutabile dei frammenti."""

import pytest

from memory_fragments.archive.static import StaticArchive
from memory_fragments.models import Fragment, FragmentConditions, FragmentMetadata, FragmentStatus


def _make_fragment(
    fragment_id: str = "F-0001",
    content: str = "Il gatto dorme sul divano.",
    quality: float = 0.8,
    topic: str = "animali",
    tags=None,
) -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        content=content,
        metadata=FragmentMetadata(topic=topic, quality=quality, tags=tags or []),
        conditions=FragmentConditions(keywords=["gatto", "divano"]),
    )


class TestAdd:
    def test_add_and_get(self):
        archive = StaticArchive()
        archive.add(_make_fragment())
        assert archive.count() == 1
        frag = archive.get("F-0001")
        assert frag is not None
        assert frag.content == "Il gatto dorme sul divano."

    def test_get_returns_deep_copy(self):
        archive = StaticArchive()
        archive.add(_make_fragment())
        first = archive.get("F-0001")
        first.content = "MUTATO"
        second = archive.get("F-0001")
        assert second.content == "Il gatto dorme sul divano."

    def test_duplicate_id_raises(self):
        archive = StaticArchive()
        archive.add(_make_fragment())
        with pytest.raises(ValueError):
            archive.add(_make_fragment())

    def test_empty_checksum_raises(self):
        archive = StaticArchive()
        frag = _make_fragment()
        frag.checksum = ""
        with pytest.raises(ValueError):
            archive.add(frag)

    def test_content_too_long_raises(self):
        archive = StaticArchive()
        frag = _make_fragment(content="x" * 10_001)
        with pytest.raises(ValueError):
            archive.add(frag)

    def test_missing_returns_none(self):
        archive = StaticArchive()
        assert archive.get("NOPE") is None


class TestDelete:
    def test_soft_delete(self):
        archive = StaticArchive()
        archive.add(_make_fragment())
        assert archive.delete("F-0001") is True
        assert archive.get("F-0001").status == FragmentStatus.ARCHIVED

    def test_delete_missing_returns_false(self):
        assert StaticArchive().delete("NOPE") is False


class TestListSearch:
    def _populated(self) -> StaticArchive:
        archive = StaticArchive()
        archive.add(_make_fragment("F-1", "il gatto dorme", topic="animali", tags=["gatto"]))
        archive.add(_make_fragment("F-2", "il cane abbaia", quality=0.9, topic="animali", tags=["cane"]))
        archive.add(_make_fragment("F-3", "la pianta cresce", quality=0.5, topic="botanica", tags=["pianta"]))
        return archive

    def test_list_by_topic(self):
        results = self._populated().list(topic="animali")
        assert {f.fragment_id for f in results} == {"F-1", "F-2"}

    def test_list_by_tag_case_insensitive(self):
        results = self._populated().list(tags=["GATTO"])
        assert [f.fragment_id for f in results] == ["F-1"]

    def test_list_by_status(self):
        archive = self._populated()
        archive.delete("F-1")
        results = archive.list(status=FragmentStatus.ARCHIVED)
        assert [f.fragment_id for f in results] == ["F-1"]

    def test_search_by_keywords_case_insensitive(self):
        results = self._populated().search_by_keywords(["CANE"])
        assert [f.fragment_id for f in results] == ["F-2"]

    def test_search_by_quality(self):
        results = self._populated().search_by_quality(0.8)
        assert {f.fragment_id for f in results} == {"F-1", "F-2"}

    def test_search_empty_keywords(self):
        assert self._populated().search_by_keywords([]) == []


class TestSerialization:
    def test_round_trip(self):
        archive = StaticArchive()
        archive.add(_make_fragment("F-1"))
        archive.add(_make_fragment("F-2", "il cane abbaia"))
        restored = StaticArchive.from_dict(archive.to_dict())
        assert len(restored) == 2
        assert restored.get("F-2").content == "il cane abbaia"

    def test_len_contains(self):
        archive = StaticArchive()
        archive.add(_make_fragment("F-1"))
        assert len(archive) == 1
        assert "F-1" in archive
        assert "F-9" not in archive

    def test_all_fragment_ids(self):
        archive = StaticArchive()
        archive.add(_make_fragment("F-1"))
        archive.add(_make_fragment("F-2", "il cane abbaia"))
        assert set(archive.all_fragment_ids()) == {"F-1", "F-2"}
