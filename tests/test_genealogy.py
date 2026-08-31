"""Tests for the GenealogyGraph (DAG versioning)."""

from memory_fragments.models import GenealogyGraph


class TestAddNode:
    def test_root_depth_zero(self):
        graph = GenealogyGraph()
        graph.add_node("root-1")
        assert graph.get_node("root-1").depth == 0
        assert graph.node_count() == 1

    def test_child_depth_from_parent(self):
        graph = GenealogyGraph()
        graph.add_node("root-1")
        graph.add_node("child-1", parent_ids=["root-1"])
        graph.add_node("grandchild-1", parent_ids=["child-1"])

        assert graph.get_node("root-1").depth == 0
        assert graph.get_node("child-1").depth == 1
        assert graph.get_node("grandchild-1").depth == 2

    def test_child_pointers_updated(self):
        graph = GenealogyGraph()
        graph.add_node("root-1")
        graph.add_node("child-1", parent_ids=["root-1"])
        assert "child-1" in graph.get_node("root-1").child_ids

    def test_remove_node(self):
        graph = GenealogyGraph()
        graph.add_node("root-1")
        assert graph.remove_node("root-1") is True
        assert graph.remove_node("root-1") is False


class TestTraversal:
    def test_ancestors(self):
        graph = GenealogyGraph()
        graph.add_node("r")
        graph.add_node("c1", parent_ids=["r"])
        graph.add_node("c2", parent_ids=["c1"])
        ancestors = graph.get_ancestors("c2")
        ids = {n.fragment_id for n in ancestors}
        assert ids == {"c2", "c1", "r"}

    def test_descendants(self):
        graph = GenealogyGraph()
        graph.add_node("r")
        graph.add_node("c1", parent_ids=["r"])
        graph.add_node("c2", parent_ids=["r"])
        graph.add_node("c3", parent_ids=["c1"])
        descendants = graph.get_descendants("r")
        ids = {n.fragment_id for n in descendants}
        assert ids == {"r", "c1", "c2", "c3"}

    def test_lineage_ordered_root_to_node(self):
        graph = GenealogyGraph()
        graph.add_node("r")
        graph.add_node("c1", parent_ids=["r"])
        graph.add_node("c2", parent_ids=["c1"])
        lineage = graph.get_lineage("c2")
        assert lineage == ["r", "c1", "c2"]

    def test_roots_and_leaves(self):
        graph = GenealogyGraph()
        graph.add_node("r")
        graph.add_node("c1", parent_ids=["r"])
        graph.add_node("c2", parent_ids=["r"])
        assert [n.fragment_id for n in graph.get_roots()] == ["r"]
        assert {n.fragment_id for n in graph.get_leaves()} == {"c1", "c2"}

    def test_no_cycles(self):
        graph = GenealogyGraph()
        graph.add_node("a")
        graph.add_node("b", parent_ids=["a"])
        graph.add_node("c", parent_ids=["b"])
        assert graph.detect_cycles() == []


class TestModelProvenance:
    def test_nodes_by_model(self):
        graph = GenealogyGraph()
        graph.add_node("m1", model_id="gpt-4o")
        graph.add_node("m2", model_id="phi-3-mini")
        assert [n.fragment_id for n in graph.get_nodes_by_model("gpt-4o")] == ["m1"]
        assert [n.fragment_id for n in graph.get_nodes_by_model("phi-3-mini")] == ["m2"]

    def test_model_statistics(self):
        graph = GenealogyGraph()
        graph.add_node("m1", model_id="gpt-4o", evaluator_model_id="eval-1", quality_source="majority_vote")
        graph.add_node("m2", model_id="gpt-4o", evaluator_model_id="eval-1", quality_source="majority_vote")
        stats = graph.get_model_statistics()
        assert stats["nodes_by_model"] == {"gpt-4o": 2}
        assert stats["nodes_by_evaluator"] == {"eval-1": 2}
        assert stats["nodes_by_quality_source"] == {"majority_vote": 2}
        assert stats["total_nodes"] == 2


class TestSerialization:
    def test_roundtrip(self):
        graph = GenealogyGraph()
        graph.add_node("r")
        graph.add_node("c1", parent_ids=["r"], appeal_id="app-1", approver="test-user")

        restored = GenealogyGraph.from_dict(graph.to_dict())

        assert restored.node_count() == 2
        node = restored.get_node("c1")
        assert node.parent_ids == ["r"]
        assert node.appeal_id == "app-1"
        assert node.approver == "test-user"
        assert node.depth == 1
