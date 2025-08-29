"""
Integration tests for graph traversal - tests with real file I/O.
"""

from pathlib import Path

from kg.core.graph import GraphTraverser


class TestGraphTraverserIntegration:
    """Integration tests for GraphTraverser with real files."""

    def test_load_graph(self, temp_kg_dir: Path, sample_nodes: dict):
        """Test loading graph from real files."""
        traverser = GraphTraverser(temp_kg_dir)

        # Check all nodes are loaded
        assert len(traverser.nodes) == len(sample_nodes)
        assert "node-1" in traverser.nodes
        assert "node-2" in traverser.nodes
        assert "node-3" in traverser.nodes

        # Check node data
        node1 = traverser.nodes["node-1"]
        assert node1["id"] == "node-1"
        assert "python" in node1["tags"]
        assert "testing" in node1["tags"]

    def test_traverse_real_graph(self, temp_kg_dir: Path, sample_nodes: dict):
        """Test traversing a real graph loaded from files."""
        traverser = GraphTraverser(temp_kg_dir)

        # Traverse from node-1
        result = traverser.bfs_traverse("node-1", max_depth=2, direction="out")

        # Should find connected nodes
        assert 0 in result
        assert 1 in result
        assert "node-2" in result[1]

    def test_find_path_real_graph(self, temp_kg_dir: Path, sample_nodes: dict):
        """Test finding path in real graph."""
        traverser = GraphTraverser(temp_kg_dir)

        # Find path from node-1 to node-3
        path = traverser.find_path("node-1", "node-3")

        # Should find path through node-2
        assert path is not None
        assert len(path) == 3
        assert path[0] == "node-1"
        assert path[1] == "node-2"
        assert path[2] == "node-3"
