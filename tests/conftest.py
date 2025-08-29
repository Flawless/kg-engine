"""
Pytest configuration and shared fixtures for kg tests.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def shared_model():
    """Share the embedding model across all tests in the session."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@pytest.fixture
def temp_kg_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for kg testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kg_path = Path(tmpdir)
        nodes_dir = kg_path / "nodes"
        nodes_dir.mkdir()
        yield kg_path


@pytest.fixture
def sample_nodes(temp_kg_dir: Path) -> dict:
    """Create sample nodes for testing."""
    nodes_dir = temp_kg_dir / "nodes"

    nodes = {
        "node-1": {
            "id": "node-1",
            "date": "2024-01-01",
            "tags": ["python", "testing", "kg"],
            "links": ["node:node-2", "project:test-project"],
            "title": "Test Node 1",
            "content": "This is a test node about Python testing.",
        },
        "node-2": {
            "id": "node-2",
            "date": "2024-01-02",
            "tags": ["api", "rest", "integration"],
            "links": ["node:node-3"],
            "title": "API Design",
            "content": "REST API design patterns and best practices.",
        },
        "node-3": {
            "id": "node-3",
            "date": "2024-01-03",
            "tags": ["database", "lancedb", "vector"],
            "links": ["node:node-1"],
            "title": "Vector Database",
            "content": "Using LanceDB for vector search and embeddings.",
        },
        "daily-2024-01-01": {
            "id": "daily-2024-01-01",
            "date": "2024-01-01",
            "tags": ["daily"],
            "links": ["node:node-1", "sprint:2024-01-01"],
            "title": "Daily Notes",
            "content": "Today worked on setting up the knowledge graph system.",
        },
        "sprint-2024-01-01": {
            "id": "sprint-2024-01-01",
            "date": "2024-01-01",
            "tags": ["sprint", "planning"],
            "links": [],
            "title": "Sprint Planning",
            "content": "Sprint goals: Set up kg system, implement search.",
        },
    }

    # Write nodes to files
    for node_id, node_data in nodes.items():
        file_path = nodes_dir / f"{node_id}.md"

        # Create frontmatter
        frontmatter = {
            "id": node_data["id"],
            "date": node_data["date"],
            "tags": node_data["tags"],
            "links": node_data["links"],
        }

        # Write file
        content = f"""---
{yaml.dump(frontmatter, default_flow_style=False)}---

# {node_data["title"]}

{node_data["content"]}
"""
        file_path.write_text(content)

    return nodes


@pytest.fixture
def temp_lancedb_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for LanceDB testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def cached_index_dir() -> Generator[Path, None, None]:
    """Create a cached index directory that persists across tests in a module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Monkey-patch to use shared model
def mock_sentence_transformer(model_name):
    """Return the shared model instance instead of loading a new one."""
    if not hasattr(mock_sentence_transformer, "_model"):
        print(f"Loading embedding model: {model_name}")
        mock_sentence_transformer._model = SentenceTransformer(model_name)
    return mock_sentence_transformer._model
