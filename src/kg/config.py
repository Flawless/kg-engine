"""Configuration management for KG engine."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration for KG engine."""

    def __init__(self, kg_dir: Optional[Path] = None):
        """Initialize configuration.

        Args:
            kg_dir: Path to the knowledge graph directory. If None, will try to detect.
        """
        if kg_dir:
            self.kg_dir = Path(kg_dir).resolve()
        else:
            self.kg_dir = self._detect_kg_dir()

        # Core paths
        self.nodes_dir = self.kg_dir / "nodes"
        self.index_dir = self.kg_dir / ".claude" / "lancedb"

        # Ensure directories exist
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _detect_kg_dir(self) -> Path:
        """Detect the knowledge graph directory.

        Looks for a 'nodes' directory in:
        1. Current directory
        2. Parent directory
        3. KG_DIR environment variable
        """
        # Check current directory
        current = Path.cwd()
        if (current / "nodes").exists():
            return current

        # Check parent directory
        parent = current.parent
        if (parent / "nodes").exists():
            return parent

        # Check environment variable
        if kg_dir_env := os.environ.get("KG_DIR"):
            kg_dir = Path(kg_dir_env)
            if (kg_dir / "nodes").exists():
                return kg_dir

        # Default to current directory (will create nodes/ if needed)
        return current

    @classmethod
    def get_instance(cls) -> "Config":
        """Get or create singleton config instance."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


# Global config instance
config = Config.get_instance()
