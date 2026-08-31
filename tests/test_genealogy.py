"""Test della GenealogyGraph — DAG di versioning e provenienza."""

from memory_fragments.models.graph import GenealogyGraph


class TestMutation:
    def test_add_node_root(self):
        g = GenealogyGraph()
        node = g.add_node("F-1")
        assert node.depth == 0
        assert g.node_count() == 1
        assert g.get_node("F-1") is not None

    def test_depth_from_parents(self):
        g = GenealogyGraph()
        g.add_node("F-1")
        g.add_node("F-2", parent_ids=["F-1"])
        node = g.add_node("F-3", parent_ids=["F-2"])
        assert node.depth == 2

    def test_child_pointers(self):
        g = GenealogyGraph()
        g.add_node("F-1")
        g.add_node("F-2", parent_ids=["F-1"])
        assert "F-2" in g.get_node("F-1").child_ids

    def test_remove_node(self):
        g = GenealogyGraph()
        g.add_node("F-1")
        assert g.remove_node("F-1") is True
        assert g.remove_node("F-1") is False
        assert g.node_count() == 0


class TestTraversal:
    def _chain(self) -> GenealogyGraph:
        g = GenealogyGraph()
        g.add_node("F-1")
        g.add_node("F-2", parent_ids=["F-1"])
        g.add_node("F-3", parent_ids=["F-2"])
        return g

    def test_ancestors(self):
        g = self._chain()
        ids = {n.fragment_id for n in g.get_ancestors("F-3")}
        assert ids == {"F-1", "F-2", "F-3"}

    def test_descendants(self):
        g = self._chain()
        ids = {n.fragment_id for n in g.get_descendants("F-1")}
        assert ids == {"F-1", "F-2", "F-3"}

    def test_lineage(self):
        g = self._chain()
        assert g.get_lineage("F-3") == ["F-1", "F-2", "F-3"]

    def test_roots_and_leaves(self):
        g = self._chain()
        assert [n.fragment_id for n in g.get_roots()] == ["F-1"]
        assert [n.fragment_id for n in g.get_leaves()] == ["F-3"]


class TestIntegrity:
    def test_no_cycles_on_chain(self):
        g = GenealogyGraph()
        g.add_node("F-1")
        g.add_node("F-2", parent_ids=["F-1"])
        assert g.detect_cycles() == []

    def test_max_depth_filter(self):
        g = GenealogyGraph()
        g.add_node("F-1")
        g.add_node("F-2", parent_ids=["F-1"])
        g.add_node("F-3", parent_ids=["F-2"])
        ids = {n.fragment_id for n in g.get_ancestors("F-3", max_depth=1)}
        assert ids == {"F-2", "F-3"}


class TestModelProvenance:
    def test_nodes_by_model(self):
        g = GenealogyGraph()
        g.add_node("F-1", model_id="gpt-4o", model_version="1.0")
        g.add_node("F-2", model_id="gpt-4o", model_version="1.0")
        g.add_node("F-3", model_id="phi-3")
        assert len(g.get_nodes_by_model("gpt-4o")) == 2

    def test_statistics(self):
        g = GenealogyGraph()
        g.add_node("F-1", model_id="gpt-4o", quality_source="majority_vote")
        g.add_node("F-2", model_id="phi-3", quality_source="user_claimed")
        stats = g.get_model_statistics()
        assert stats["total_nodes"] == 2
        assert stats["nodes_by_model"]["gpt-4o"] == 1


class TestSerialization:
    def test_round_trip(self):
        g = GenealogyGraph()
        g.add_node("F-1", model_id="gpt-4o")
        g.add_node("F-2", parent_ids=["F-1"], appeal_id="A-1", checksum="abc")
        restored = GenealogyGraph.from_dict(g.to_dict())
        assert restored.node_count() == 2
        node = restored.get_node("F-2")
        assert node.parent_ids == ["F-1"]
        assert node.appeal_id == "A-1"
        assert node.checksum == "abc"

    def test_to_json(self):
        import json

        g = GenealogyGraph()
        g.add_node("F-1")
        data = json.loads(g.to_json())
        assert "F-1" in data["nodes"]
