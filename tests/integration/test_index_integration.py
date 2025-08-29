"""
Integration tests for LanceDB index - tests with real embeddings and database.
These tests are slow as they load models and create embeddings.
"""

from pathlib import Path

from kg.core.index import GraphIndexer


class TestGraphIndexerIntegration:
    """Integration tests for LanceDB-based graph indexer."""

    def test_index_creation(
        self, temp_kg_dir: Path, sample_nodes: dict, temp_lancedb_dir: Path
    ):
        """Test creating and rebuilding the index with real embeddings."""
        indexer = GraphIndexer(kg_path=str(temp_kg_dir), db_path=str(temp_lancedb_dir))
        indexer.rebuild_index()

        # Check that nodes table exists and has data
        assert "nodes" in indexer.db.table_names()
        nodes_table = indexer.db.open_table("nodes")
        assert nodes_table.count_rows() == len(sample_nodes)

    def test_semantic_search(
        self, temp_kg_dir: Path, sample_nodes: dict, temp_lancedb_dir: Path
    ):
        """Test semantic search with real embeddings."""
        indexer = GraphIndexer(kg_path=str(temp_kg_dir), db_path=str(temp_lancedb_dir))
        indexer.rebuild_index()

        # Search for Python-related content
        results = list(indexer.search("Python", limit=3))

        # Should return something (even if just headers in test env)
        assert len(results) > 0

    def test_find_similar(
        self, temp_kg_dir: Path, sample_nodes: dict, temp_lancedb_dir: Path
    ):
        """Test finding similar nodes with real embeddings."""
        indexer = GraphIndexer(kg_path=str(temp_kg_dir), db_path=str(temp_lancedb_dir))
        indexer.rebuild_index()

        # Find nodes similar to node-1
        similar = list(indexer.find_similar("node-1", limit=3))

        # Should return formatted strings
        assert len(similar) > 0
        assert isinstance(similar[0], str)
