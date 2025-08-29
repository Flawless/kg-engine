"""
Unit tests for graph traversal functionality - fast, no external dependencies.
"""

from pathlib import Path
from unittest.mock import patch

from kg.core.graph import GraphTraverser


class TestGraphTraverserUnit:
    """Unit tests for GraphTraverser - fast tests without file I/O."""

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_init(self, mock_load):
        """Test GraphTraverser initialization."""
        traverser = GraphTraverser("/fake/path")
        assert traverser.kg_path == Path("/fake/path")
        assert traverser.nodes == {}
        assert traverser.edges is not None
        assert traverser.reverse_edges is not None
        mock_load.assert_called_once()

    def test_extract_frontmatter(self):
        """Test frontmatter extraction from markdown."""
        traverser = GraphTraverser(".")
        content = """---
id: test-node
date: 2024-01-01
tags: [test, example]
---

# Test Title

Some content here.
"""
        frontmatter, title = traverser.extract_frontmatter(content)
        assert frontmatter["id"] == "test-node"
        # YAML parser converts date to datetime.date object
        import datetime

        assert frontmatter["date"] == datetime.date(2024, 1, 1)
        assert "test" in frontmatter["tags"]
        assert title == "Test Title"

    def test_extract_frontmatter_no_yaml(self):
        """Test extraction when no frontmatter present."""
        traverser = GraphTraverser(".")
        content = "# Just a title\n\nSome content"
        frontmatter, title = traverser.extract_frontmatter(content)
        assert frontmatter == {}
        # When no frontmatter, the method returns empty string for title
        assert title == ""

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_bfs_traverse_empty(self, mock_load):
        """Test BFS traversal with empty graph."""
        traverser = GraphTraverser(".")
        traverser.nodes = {}
        traverser.edges = {}
        traverser.reverse_edges = {}

        result = traverser.bfs_traverse("nonexistent")
        assert result == {}

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_bfs_traverse_single_node(self, mock_load):
        """Test BFS traversal with single node."""
        traverser = GraphTraverser(".")
        traverser.nodes = {"node1": {"id": "node1", "title": "Node 1"}}
        traverser.edges = {"node1": set()}
        traverser.reverse_edges = {}

        result = traverser.bfs_traverse("node1", max_depth=1)
        assert 0 in result
        assert "node1" in result[0]
        assert 1 not in result  # No neighbors

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_bfs_traverse_with_edges(self, mock_load):
        """Test BFS traversal with edges."""
        traverser = GraphTraverser(".")
        traverser.nodes = {
            "node1": {"id": "node1"},
            "node2": {"id": "node2"},
            "node3": {"id": "node3"},
        }
        traverser.edges = {"node1": {"node2", "node3"}, "node2": {"node3"}}
        traverser.reverse_edges = {"node2": {"node1"}, "node3": {"node1", "node2"}}

        # Test outgoing traversal
        result = traverser.bfs_traverse("node1", max_depth=1, direction="out")
        assert "node2" in result[1]
        assert "node3" in result[1]

        # Test incoming traversal
        result = traverser.bfs_traverse("node3", max_depth=1, direction="in")
        assert "node1" in result[1]
        assert "node2" in result[1]

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_find_path_same_node(self, mock_load):
        """Test finding path to same node."""
        traverser = GraphTraverser(".")
        traverser.nodes = {"node1": {"id": "node1"}}

        path = traverser.find_path("node1", "node1")
        assert path == ["node1"]

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_find_path_no_connection(self, mock_load):
        """Test finding path with no connection."""
        traverser = GraphTraverser(".")
        traverser.nodes = {"node1": {"id": "node1"}, "node2": {"id": "node2"}}
        traverser.edges = {}
        traverser.reverse_edges = {}

        path = traverser.find_path("node1", "node2")
        assert path is None

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_get_node_centrality(self, mock_load):
        """Test centrality calculation."""
        traverser = GraphTraverser(".")
        traverser.nodes = {
            "node1": {"id": "node1"},
            "node2": {"id": "node2"},
            "node3": {"id": "node3"},
        }
        traverser.edges = {"node1": {"node2", "node3"}, "node2": {"node3"}}
        traverser.reverse_edges = {"node2": {"node1"}, "node3": {"node1", "node2"}}

        centrality = traverser.get_node_centrality()

        # Find node3's centrality
        # Tuple order is: (node_id, total_degree, in_degree, out_degree)
        node3_cent = next(c for c in centrality if c[0] == "node3")
        assert node3_cent[1] == 2  # total_degree (2 in + 0 out)
        assert node3_cent[2] == 2  # in_degree
        assert node3_cent[3] == 0  # out_degree

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_find_clusters_single_component(self, mock_load):
        """Test finding clusters in connected graph."""
        traverser = GraphTraverser(".")
        traverser.nodes = {
            "node1": {"id": "node1"},
            "node2": {"id": "node2"},
            "node3": {"id": "node3"},
        }
        traverser.edges = {"node1": {"node2"}, "node2": {"node3"}, "node3": {"node1"}}
        traverser.reverse_edges = {
            "node1": {"node3"},
            "node2": {"node1"},
            "node3": {"node2"},
        }

        clusters = traverser.find_clusters(min_size=1)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    @patch("kg.core.graph.GraphTraverser.load_graph")
    def test_find_clusters_disconnected(self, mock_load):
        """Test finding clusters in disconnected graph."""
        traverser = GraphTraverser(".")
        traverser.nodes = {
            "node1": {"id": "node1"},
            "node2": {"id": "node2"},
            "node3": {"id": "node3"},
            "node4": {"id": "node4"},
        }
        traverser.edges = {"node1": {"node2"}, "node3": {"node4"}}
        traverser.reverse_edges = {"node2": {"node1"}, "node4": {"node3"}}

        clusters = traverser.find_clusters(min_size=1)
        assert len(clusters) == 2
        assert {len(c) for c in clusters} == {2, 2}
