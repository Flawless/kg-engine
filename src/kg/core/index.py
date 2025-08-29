#!/usr/bin/env python3
"""
Knowledge Graph LanceDB Index

Maintains a LanceDB index with both full-text and semantic search capabilities.
"""

import os

# Suppress transformers/torch verbosity before importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import argparse
import logging
import pandas as pd
import difflib
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
import lancedb
from nltk.stem import PorterStemmer
import nltk

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables

env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config in current working directory
        if Path("kg_config.yaml").exists():
            config_path = "kg_config.yaml"
            logger.debug(f"Found config file: {config_path}")
        else:
            logger.debug("No config file found, using defaults")

    if config_path and Path(config_path).exists():
        logger.debug(f"Loading config from: {Path(config_path).resolve()}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Return default config if no file found
    logger.debug("Using default configuration")
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "search_weights": {
            "semantic": 0.55,
            "connectivity": 0.05,
            "title": 0.20,
            "tag": 0.20,
            "content": 0.0,
        },
        "connectivity": {
            "max_depth": 2,
            "decay_factor": 0.5,
            "link_weights": {"node": 1.0, "task": 0.8, "project": 0.6, "sprint": 0.4},
        },
        "fuzzy_matching": {"threshold": 0.6, "partial_threshold": 0.7},
        "paths": {"kg_path": ".", "db_path": ".claude/lancedb"},
    }


class GraphIndexer:
    # Default search scoring weights (Elisabeth's recommendations)
    DEFAULT_WEIGHTS = {
        "semantic": 0.55,  # Semantic similarity (from embeddings)
        "connectivity": 0.05,  # How connected the node is
        "title": 0.20,  # Fuzzy match in title
        "tag": 0.20,  # Fuzzy match in tags
        "content": 0.0,  # Match in content (if enabled)
    }

    # Recursive connectivity scoring parameters
    DEFAULT_CONNECTIVITY_DEPTH = 2  # Maximum recursion depth
    DEFAULT_DECAY_FACTOR = 0.5  # Score decay per level (0.5 = half strength each hop)
    DEFAULT_LINK_WEIGHTS = {
        "node": 1.0,  # Direct node references
        "task": 0.8,  # Task links (slightly less weight)
        "project": 0.6,  # Project links
        "sprint": 0.4,  # Sprint links (temporal, less structural)
    }

    # Fuzzy matching parameters
    FUZZY_THRESHOLD = 0.6  # Minimum similarity for fuzzy matches
    PARTIAL_THRESHOLD = 0.7  # Threshold for partial word matches

    def __init__(
        self,
        kg_path: str = None,
        db_path: str = None,
        model_name: str = None,
        weights: Optional[Dict[str, float]] = None,
        connectivity_depth: int = None,
        decay_factor: float = None,
        link_weights: Optional[Dict[str, float]] = None,
        config_path: str = None,
    ):
        # Load configuration
        config = load_config(config_path)

        # Debug logging
        logger.debug(f"Config path: {config_path}")
        logger.debug(f"Config kg_path: {config['paths']['kg_path']}")
        logger.debug(f"Config db_path: {config['paths']['db_path']}")
        logger.debug(f"Provided kg_path: {kg_path}")
        logger.debug(f"Current working directory: {os.getcwd()}")

        # Use provided values or fall back to config
        self.kg_path = Path(kg_path or config["paths"]["kg_path"])

        # If db_path is provided, use it; otherwise construct from config relative to kg_path
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Make db_path relative to kg_path, not current directory
            db_path_config = config["paths"]["db_path"]
            if Path(db_path_config).is_absolute():
                self.db_path = Path(db_path_config)
            else:
                self.db_path = self.kg_path / db_path_config

        # Debug logging
        logger.debug(f"Resolved kg_path: {self.kg_path.resolve()}")
        logger.debug(f"Resolved db_path: {self.db_path.resolve()}")

        self.db_path.mkdir(parents=True, exist_ok=True)

        # Set weights with validation
        self.weights = self._validate_weights(weights or config["search_weights"])

        # Set connectivity parameters
        self.connectivity_depth = (
            connectivity_depth or config["connectivity"]["max_depth"]
        )
        self.decay_factor = decay_factor or config["connectivity"]["decay_factor"]
        self.link_weights = link_weights or config["connectivity"]["link_weights"]

        # Set fuzzy matching parameters
        self.FUZZY_THRESHOLD = config["fuzzy_matching"]["threshold"]
        self.PARTIAL_THRESHOLD = config["fuzzy_matching"]["partial_threshold"]

        # Get model from parameters, env, or config
        if model_name is None:
            model_name = os.environ.get("EMBEDDING_MODEL", config["embedding_model"])

        # Initialize embedding model silently
        logger.debug(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name, device="cpu" if os.environ.get("NO_GPU") else None
        )

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.init_tables()

        # Initialize NLTK stemmer
        # Download required NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.stemmer = PorterStemmer()

    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights sum to 1.0 and contain all required keys."""
        required_keys = {"semantic", "connectivity", "title", "tag", "content"}

        # Check all required keys are present
        if not all(key in weights for key in required_keys):
            missing = required_keys - weights.keys()
            raise ValueError(f"Missing weight keys: {missing}")

        # Check weights sum to 1.0 (with small tolerance for float precision)
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

        return weights

    def init_tables(self):
        """Initialize or get existing tables."""
        # Check if nodes table exists
        existing_tables = self.db.table_names()

        if "nodes" not in existing_tables:
            # Create empty table with schema
            # Ensure both vector columns have the same embedding size
            dummy_embedding = self.model.encode("dummy", show_progress_bar=False)
            schema_data = [
                {
                    "id": "dummy",
                    "filename": "dummy.md",
                    "title": "",
                    "date": "",
                    "content": "",
                    "tags": ["dummy"],
                    "links": ["dummy"],
                    "vector": dummy_embedding,
                    "tag_vector": dummy_embedding,  # Same size as vector
                    "last_modified": datetime.now(),
                }
            ]
            self.nodes_table = self.db.create_table(
                "nodes", data=schema_data, mode="overwrite"
            )
            # Delete the dummy row
            self.nodes_table.delete("id = 'dummy'")
        else:
            self.nodes_table = self.db.open_table("nodes")

    def extract_frontmatter(self, content: str) -> tuple[Dict[str, Any], str, str]:
        """Extract YAML frontmatter and content from markdown."""
        if not content.startswith("---\n"):
            return {}, "", content

        try:
            end_marker = content.find("\n---\n", 4)
            if end_marker == -1:
                return {}, "", content

            yaml_content = content[4:end_marker]
            body = content[end_marker + 5 :]

            # Extract title from first # heading
            title = ""
            for line in body.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            frontmatter = yaml.safe_load(yaml_content) or {}
            return frontmatter, title, body
        except yaml.YAMLError as e:
            logger.warning(f"YAML parse error: {e}")
            return {}, "", content

    def index_file(self, file_path: Path):
        """Index a single markdown file."""
        if not file_path.exists() or not file_path.suffix == ".md":
            return

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        frontmatter, title, body = self.extract_frontmatter(content)

        if "id" not in frontmatter:
            logger.warning(f"Warning: {file_path} missing id field")
            return

        node_id = frontmatter["id"]
        date = frontmatter.get("date", "")
        tags = frontmatter.get("tags", [])
        links = frontmatter.get("links", [])

        # Create embeddings
        # Content embedding: title + body
        content_text = f"{title}\n{body}"
        content_embedding = self.model.encode(content_text, show_progress_bar=False)

        # Tag embedding: tags + title (tags are more important)
        tag_text = f"{' '.join(tags)} {title}" if tags else title
        tag_embedding = self.model.encode(tag_text, show_progress_bar=False)

        # Get file modification time
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

        # Prepare data for LanceDB
        # Handle relative path safely
        try:
            relative_path = str(file_path.relative_to(self.kg_path))
        except ValueError:
            # Fallback if paths don't work with relative_to
            relative_path = str(file_path.name)

        node_data = {
            "id": node_id,
            "filename": relative_path,
            "title": title,
            "date": date,
            "content": body,
            "tags": tags,
            "links": links,
            "vector": content_embedding,
            "tag_vector": tag_embedding,
            "last_modified": last_modified,
        }

        # Delete existing entry if it exists
        self.nodes_table.delete(f"id = '{node_id}'")

        # Insert new data
        self.nodes_table.add([node_data])
        logger.debug(f"Indexed: {file_path.name}")

    def rebuild_index(self):
        """Rebuild entire index from scratch."""
        logger.info("Rebuilding entire index...")

        # Recreate table
        self.db.drop_table("nodes")
        self.init_tables()

        # Index all files
        nodes_dir = self.kg_path / "nodes"
        if nodes_dir.exists():
            md_files = list(nodes_dir.glob("*.md"))
            logger.info(f"Found {len(md_files)} markdown files")

            for file_path in md_files:
                self.index_file(file_path)

        logger.info(f"Index rebuilt: {self.get_stats()}")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        df = self.nodes_table.to_pandas()

        stats = {
            "total_nodes": len(df),
            "unique_tags": len(set(tag for tags in df["tags"] for tag in tags)),
            "total_links": sum(len(links) for links in df["links"]),
            "node_links": sum(
                1 for links in df["links"] for link in links if link.startswith("node:")
            ),
            "latest_node": df.loc[df["date"].idxmax()]["id"] if not df.empty else None,
        }

        return stats

    def _simple_stem(self, word: str) -> str:
        """Stem word using NLTK Porter Stemmer."""
        word = word.lower().strip()
        return self.stemmer.stem(word)

    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tags by stemming, lowercasing, and deduplicating."""
        if not tags:
            return []

        normalized = []
        seen_stems = set()

        for tag in tags:
            # Lowercase and strip
            tag_clean = tag.lower().strip()

            # Apply stemming
            tag_stem = self._simple_stem(tag_clean)

            # Only add if we haven't seen this stem before
            if tag_stem not in seen_stems:
                seen_stems.add(tag_stem)
                normalized.append(tag_stem)

        return normalized

    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess query into stemmed terms."""
        # Split on word boundaries and clean
        words = re.findall(r"\b\w+\b", query.lower())

        # Stem each word
        stemmed = [self._simple_stem(word) for word in words]

        # Include both stemmed and original terms (for better fuzzy matching)
        all_terms = words + stemmed

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for term in all_terms:
            if term not in seen and len(term) > 1:
                seen.add(term)
                result.append(term)
        return result

    def _fuzzy_text_match(self, query_terms: List[str], text: str) -> float:
        """Calculate fuzzy match score between query terms and text."""
        if not query_terms or not text:
            return 0.0

        text_lower = text.lower()
        max_score = 0.0

        for term in query_terms:
            # Exact substring match gets highest score
            if term in text_lower:
                max_score = max(max_score, 1.0)
                continue

            # Fuzzy match with difflib for whole text
            similarity = difflib.SequenceMatcher(None, term, text_lower).ratio()
            if similarity >= self.FUZZY_THRESHOLD:
                max_score = max(
                    max_score, similarity * 0.9
                )  # Slightly lower than exact
                continue

            # Check for partial matches in words
            words = re.findall(r"\b\w+\b", text_lower)
            for word in words:
                # Stemmed comparison
                word_stem = self._simple_stem(word)
                if term == word_stem:
                    max_score = max(max_score, 0.8)
                    continue

                # Fuzzy match individual words
                word_similarity = difflib.SequenceMatcher(None, term, word).ratio()
                if word_similarity >= self.PARTIAL_THRESHOLD:
                    max_score = max(max_score, word_similarity * 0.7)

        return max_score

    def _get_link_weight(self, link: str) -> float:
        """Get weight for a specific link type."""
        if ":" not in link:
            return 0.0

        link_type = link.split(":", 1)[0]
        return self.link_weights.get(
            link_type, 0.5
        )  # Default weight for unknown link types

    def _calculate_recursive_connectivity(
        self, all_nodes_df: pd.DataFrame, semantic_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate recursive connectivity scores for all nodes using weighted propagation.

        Args:
            all_nodes_df: DataFrame containing all nodes
            semantic_scores: Dict mapping node_id to its semantic score for current query

        Returns:
            Dict mapping node_id to its recursive connectivity score
        """
        # Build adjacency list for efficient traversal
        incoming_links = {}  # node_id -> [(source_node_id, link_weight), ...]

        for _, row in all_nodes_df.iterrows():
            node_id = row["id"]
            if node_id not in incoming_links:
                incoming_links[node_id] = []

            # Process outgoing links from this node
            for link in row["links"]:
                if isinstance(link, str) and ":" in link:
                    target_id = link.split(":", 1)[1]
                    if target_id in all_nodes_df["id"].values:
                        link_weight = self._get_link_weight(link)
                        if target_id not in incoming_links:
                            incoming_links[target_id] = []
                        incoming_links[target_id].append((node_id, link_weight))

        # Cache for memoization: (node_id, depth) -> score
        score_cache = {}

        def compute_connectivity_score(node_id: str, depth: int, visited: set) -> float:
            """Recursively compute connectivity score for a node."""
            # Base cases
            if depth <= 0:
                return 0.0

            if node_id in visited:  # Cycle prevention
                return 0.0

            # Check cache
            cache_key = (node_id, depth)
            if cache_key in score_cache:
                return score_cache[cache_key]

            score = 0.0
            visited_with_current = visited | {node_id}

            # Process each incoming link
            for source_id, link_weight in incoming_links.get(node_id, []):
                if source_id in visited:  # Skip if would create cycle
                    continue

                # Get semantic relevance of the source node
                source_semantic_score = semantic_scores.get(source_id, 0.0)

                # Direct contribution: link_weight * source_relevance * decay
                direct_contribution = (
                    link_weight * source_semantic_score * self.decay_factor
                )
                score += direct_contribution

                # Recursive contribution from source's connections
                if depth > 1:
                    recursive_contribution = (
                        compute_connectivity_score(
                            source_id, depth - 1, visited_with_current
                        )
                        * self.decay_factor
                    )
                    score += recursive_contribution

            # Cache the result
            score_cache[cache_key] = score
            return score

        # Calculate scores for all nodes
        connectivity_scores = {}
        for node_id in all_nodes_df["id"]:
            connectivity_scores[node_id] = compute_connectivity_score(
                node_id, self.connectivity_depth, set()
            )

        return connectivity_scores

    def _calculate_tag_semantic_score(self, query_embedding, tags) -> float:
        """Calculate semantic similarity by comparing query to each tag individually."""
        # Handle both list and numpy array
        if isinstance(tags, np.ndarray):
            tags = tags.tolist()
        elif not isinstance(tags, list):
            return 0.0

        if len(tags) == 0:
            return 0.0

        # Normalize tags
        normalized_tags = self._normalize_tags(tags)

        # Get embeddings for each normalized tag
        tag_scores = []
        for tag in normalized_tags:
            tag_embedding = self.model.encode(tag, show_progress_bar=False)
            # Calculate cosine similarity manually
            # Normalize embeddings first
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            tag_norm = tag_embedding / np.linalg.norm(tag_embedding)
            similarity = float(query_norm @ tag_norm)
            # Convert to 0-1 range (cosine similarity is -1 to 1, but usually positive for text)
            similarity = max(0.0, similarity)
            tag_scores.append(similarity)

        # Return average similarity across all tags
        return sum(tag_scores) / len(tag_scores) if tag_scores else 0.0

    def search(
        self, query: str, limit: int = 10, include_content: bool = False
    ) -> pd.DataFrame:
        """Combined search using semantic similarity, fuzzy text matching, and graph connectivity.

        Args:
            query: Search query
            limit: Max results
            include_content: If True, also search in content (default: tags only)
        """
        query_lower = query.lower()
        query_embedding = self.model.encode(query, show_progress_bar=False)

        # Get all nodes for graph analysis
        all_nodes_df = self.nodes_table.to_pandas()

        # Use tag_vector for tag-only search, content vector for content search
        if include_content:
            results = (
                self.nodes_table.search(query_embedding).limit(limit * 3).to_pandas()
            )
        else:
            # For tag-only search, we'll use tag_vector if available, otherwise fall back to regular vector
            try:
                results = (
                    self.nodes_table.search(
                        query_embedding, vector_column_name="tag_vector"
                    )
                    .limit(limit * 3)
                    .to_pandas()
                )
            except Exception:
                # Fallback to regular vector column if tag_vector fails
                results = (
                    self.nodes_table.search(query_embedding)
                    .limit(limit * 3)
                    .to_pandas()
                )

        # Calculate scoring components
        # Use individual tag semantic scoring for better tag-to-query matching
        if not include_content:
            # For tag-focused search, use individual tag embeddings
            results["semantic_score"] = results["tags"].apply(
                lambda tags: self._calculate_tag_semantic_score(query_embedding, tags)
            )
        else:
            # For content search, use the pre-computed vector similarity
            results["semantic_score"] = 1 / (
                1 + results["_distance"]
            )  # Convert distance to similarity

        # Build semantic scores dict for all nodes (needed for recursive connectivity)
        semantic_scores = dict(zip(results["id"], results["semantic_score"]))
        # Fill in missing nodes with zero semantic score
        for node_id in all_nodes_df["id"]:
            if node_id not in semantic_scores:
                semantic_scores[node_id] = 0.0

        # Calculate recursive connectivity scores
        connectivity_scores = self._calculate_recursive_connectivity(
            all_nodes_df, semantic_scores
        )

        # Normalize connectivity scores using a softer approach
        # Instead of dividing by max (which always gives someone 1.0),
        # use a sigmoid-like function to map scores to [0, 1] range
        # This prevents any single node from dominating with conn:1.0
        if connectivity_scores:
            # Use 95th percentile as a soft upper bound for better normalization
            scores_list = list(connectivity_scores.values())
            scores_list.sort()
            percentile_95 = (
                scores_list[int(len(scores_list) * 0.95)]
                if len(scores_list) > 1
                else scores_list[0]
            )

            # If 95th percentile is 0, use mean + 2*std as normalization factor
            if percentile_95 == 0:
                mean_score = sum(scores_list) / len(scores_list)
                std_score = (
                    sum((x - mean_score) ** 2 for x in scores_list) / len(scores_list)
                ) ** 0.5
                norm_factor = (
                    mean_score + 2 * std_score
                    if mean_score + 2 * std_score > 0
                    else 1.0
                )
            else:
                norm_factor = (
                    percentile_95 * 2
                )  # Multiply by 2 so 95th percentile maps to ~0.5

            # Apply sigmoid-like normalization: score / (score + norm_factor)
            normalized_connectivity = {
                k: v / (v + norm_factor) if norm_factor > 0 else 0.0
                for k, v in connectivity_scores.items()
            }
        else:
            normalized_connectivity = {}

        results["connectivity_score"] = results["id"].apply(
            lambda node_id: normalized_connectivity.get(node_id, 0.0)
        )

        # Preprocess query for fuzzy matching
        query_terms = self._preprocess_query(query)

        # Fuzzy text matching scores
        results["title_match"] = results["title"].apply(
            lambda title: self._fuzzy_text_match(query_terms, title)
        )

        # Individual tag scoring - average similarity across all tags
        def score_tags_individually(tags):
            # Handle both list and numpy array
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()
            elif not isinstance(tags, list):
                return 0.0

            if len(tags) == 0:
                return 0.0

            # Normalize tags for comparison
            normalized_tags = self._normalize_tags(tags)

            # Calculate similarity for each tag individually
            tag_scores = []
            for tag in normalized_tags:
                score = self._fuzzy_text_match(query_terms, tag)
                tag_scores.append(score)

            # Return average score (sum normalized by number of tags)
            return sum(tag_scores) / len(tag_scores) if tag_scores else 0.0

        results["tag_match"] = results["tags"].apply(score_tags_individually)

        if include_content:
            results["content_match"] = results["content"].apply(
                lambda x: 0.5 if query_lower in x.lower() else 0.0
            )
        else:
            results["content_match"] = 0.0

        # Combined score with weights
        results["combined_score"] = (
            self.weights["semantic"] * results["semantic_score"]
            + self.weights["connectivity"] * results["connectivity_score"]
            + self.weights["title"] * results["title_match"]
            + self.weights["tag"] * results["tag_match"]
            + self.weights["content"] * results["content_match"]
        )

        # Sort by combined score and limit
        results = results.sort_values("combined_score", ascending=False).head(limit)

        # Add debug info to distance column
        results["_distance"] = results.apply(
            lambda row: f"{1 - row['combined_score']:.3f} (sem:{row['semantic_score']:.2f}, conn:{row['connectivity_score']:.2f}, txt:{row['title_match'] + row['tag_match']:.1f})",
            axis=1,
        )

        return results[["id", "title", "date", "tags", "_distance"]]

    def find_similar(
        self, node_id: str, limit: int = 5, use_tags: bool = True
    ) -> pd.DataFrame:
        """Find nodes similar to a given node."""
        # Get the node
        node = self.nodes_table.search().where(f"id = '{node_id}'").limit(1).to_pandas()

        if node.empty:
            print(f"Node {node_id} not found")
            return pd.DataFrame()

        # Use tag vector for similarity if requested (usually better for finding related concepts)
        vector_column = "tag_vector" if use_tags else "vector"
        vector = node.iloc[0][vector_column]

        # Search for similar nodes (excluding the node itself)
        if use_tags:
            try:
                results = (
                    self.nodes_table.search(vector, vector_column_name=vector_column)
                    .where(f"id != '{node_id}'")
                    .limit(limit)
                    .to_pandas()
                )
            except Exception:
                # Fallback to regular vector column
                results = (
                    self.nodes_table.search(vector)
                    .where(f"id != '{node_id}'")
                    .limit(limit)
                    .to_pandas()
                )
        else:
            results = (
                self.nodes_table.search(vector)
                .where(f"id != '{node_id}'")
                .limit(limit)
                .to_pandas()
            )

        return results[["id", "title", "date", "tags", "_distance"]]

    def get_backlinks(self, node_id: str) -> List[Dict[str, str]]:
        """Get all nodes that link to a given node."""
        df = self.nodes_table.to_pandas()

        backlinks = []
        for _, row in df.iterrows():
            if any(link == f"node:{node_id}" for link in row["links"]):
                backlinks.append(
                    {"id": row["id"], "title": row["title"], "date": row["date"]}
                )

        return backlinks

    def get_by_tag(self, tag: str) -> pd.DataFrame:
        """Get all nodes with a specific tag."""
        df = (
            self.nodes_table.search()
            .where(f"array_contains(tags, '{tag}')")
            .to_pandas()
        )
        return df[["id", "title", "date", "tags"]]

    def find_orphans(self) -> List[Dict[str, str]]:
        """Find nodes with no incoming links."""
        df = self.nodes_table.to_pandas()

        # Collect all linked node IDs
        linked_ids = set()
        for links in df["links"]:
            for link in links:
                if link.startswith("node:"):
                    linked_ids.add(link[5:])

        # Find nodes not in linked set
        orphans = []
        for _, row in df.iterrows():
            if row["id"] not in linked_ids:
                orphans.append(
                    {
                        "id": row["id"],
                        "filename": row["filename"],
                        "title": row["title"],
                    }
                )

        return orphans


def main(kg_path=None, config_path=None):
    # If kg_path not provided, check environment variable
    if kg_path is None:
        kg_path = os.environ.get("KG_PATH")

    # If config_path not provided, check environment variable
    if config_path is None:
        config_path = os.environ.get("KG_CONFIG")

    parser = argparse.ArgumentParser(description="Knowledge Graph LanceDB Indexer")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild entire index")
    parser.add_argument("--update", help="Update index for specific file")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--search", help="Search query")
    parser.add_argument(
        "--content",
        action="store_true",
        help="Include content in search (default: tags only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed search results with links/backlinks",
    )
    parser.add_argument("--similar", help="Find nodes similar to given node ID")
    parser.add_argument("--backlinks", help="Get backlinks for a node")
    parser.add_argument("--tag", help="Get nodes by tag")
    parser.add_argument("--orphans", action="store_true", help="Find orphaned nodes")
    parser.add_argument(
        "--list-tags", action="store_true", help="List all available tags"
    )
    parser.add_argument("--similar-tags", help="Find tags similar to given tag")
    parser.add_argument("--limit", type=int, default=5, help="Result limit")

    # Weight arguments (Elisabeth's recommended defaults)
    parser.add_argument(
        "--weight-semantic",
        type=float,
        default=0.55,
        help="Weight for semantic similarity (default: 0.55)",
    )
    parser.add_argument(
        "--weight-connectivity",
        type=float,
        default=0.05,
        help="Weight for node connectivity (default: 0.05)",
    )
    parser.add_argument(
        "--weight-title",
        type=float,
        default=0.20,
        help="Weight for title matching (default: 0.20)",
    )
    parser.add_argument(
        "--weight-tag",
        type=float,
        default=0.20,
        help="Weight for tag matching (default: 0.20)",
    )
    parser.add_argument(
        "--weight-content",
        type=float,
        default=0.0,
        help="Weight for content matching (default: 0.0)",
    )

    # Connectivity scoring parameters
    parser.add_argument(
        "--connectivity-depth",
        type=int,
        default=2,
        help="Maximum depth for recursive connectivity scoring (default: 2)",
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.5,
        help="Score decay factor per hop in connectivity (default: 0.5)",
    )
    parser.add_argument(
        "--link-weight-node",
        type=float,
        default=1.0,
        help="Weight for node: links (default: 1.0)",
    )
    parser.add_argument(
        "--link-weight-task",
        type=float,
        default=0.8,
        help="Weight for task: links (default: 0.8)",
    )
    parser.add_argument(
        "--link-weight-project",
        type=float,
        default=0.6,
        help="Weight for project: links (default: 0.6)",
    )
    parser.add_argument(
        "--link-weight-sprint",
        type=float,
        default=0.4,
        help="Weight for sprint: links (default: 0.4)",
    )

    args = parser.parse_args()

    # Build weights dictionary from arguments
    weights = {
        "semantic": args.weight_semantic,
        "connectivity": args.weight_connectivity,
        "title": args.weight_title,
        "tag": args.weight_tag,
        "content": args.weight_content,
    }

    # Build link weights dictionary from arguments
    link_weights = {
        "node": args.link_weight_node,
        "task": args.link_weight_task,
        "project": args.link_weight_project,
        "sprint": args.link_weight_sprint,
    }

    try:
        indexer = GraphIndexer(
            kg_path=kg_path,
            config_path=config_path,
            weights=weights,
            connectivity_depth=args.connectivity_depth,
            decay_factor=args.decay_factor,
            link_weights=link_weights,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.rebuild:
        indexer.rebuild_index()
    elif args.update:
        indexer.index_file(Path(args.update))
    elif args.stats:
        stats = indexer.get_stats()
        print("Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    elif args.search:
        results = indexer.search(
            args.search, limit=args.limit, include_content=args.content
        )
        search_type = "tags + content" if args.content else "tags only"

        if args.verbose:
            # Verbose format with links and backlinks (for agents)
            print(f"\nSearch results for '{args.search}' ({search_type}):")

            # Get all nodes for link information
            all_nodes_df = indexer.nodes_table.to_pandas()

            for _, row in results.iterrows():
                node_id = row["id"]
                print(f"\n- {node_id} ({row['date']})")
                print(f"  Title: {row['title']}")
                print(f"  Tags: {', '.join(row['tags'])}")
                print(f"  Score: {row['_distance']}")

                # Get direct links (outgoing)
                node_data = all_nodes_df[all_nodes_df["id"] == node_id]
                if not node_data.empty:
                    links = node_data.iloc[0]["links"]
                    # Handle numpy array
                    if isinstance(links, np.ndarray):
                        links = links.tolist()
                    if links and len(links) > 0:
                        # Format links nicely
                        formatted_links = []
                        for link in links[:5]:  # Show first 5 links
                            if ":" in link:
                                link_type, link_target = link.split(":", 1)
                                formatted_links.append(f"{link_type}:{link_target}")
                            else:
                                formatted_links.append(link)

                        links_str = ", ".join(formatted_links)
                        if len(links) > 5:
                            links_str += f" ... (+{len(links) - 5} more)"
                        print(f"  Links: {links_str}")

                # Get backlinks (incoming)
                backlinks = indexer.get_backlinks(node_id)
                if backlinks:
                    backlink_ids = [
                        bl["id"] for bl in backlinks[:5]
                    ]  # Show first 5 backlinks
                    backlinks_str = ", ".join(backlink_ids)
                    if len(backlinks) > 5:
                        backlinks_str += f" ... (+{len(backlinks) - 5} more)"
                    print(f"  Backlinks: {backlinks_str}")
        else:
            # Compact format (default for human use)
            print(f"Search '{args.search}' ({search_type}, {len(results)} results):")
            for _, row in results.iterrows():
                # Truncate tags if too many
                tags = row["tags"]
                if len(tags) > 4:
                    tags_str = f"{', '.join(tags[:4])}... (+{len(tags) - 4})"
                else:
                    tags_str = ", ".join(tags)
                print(f"  {row['id']:<30} {row['_distance']:<25} [{tags_str}]")
    elif args.similar:
        results = indexer.find_similar(args.similar, limit=args.limit)
        print(f"\nNodes similar to '{args.similar}':")
        for _, row in results.iterrows():
            print(f"- {row['id']}: {row['title']} (distance: {row['_distance']:.3f})")
    elif args.backlinks:
        backlinks = indexer.get_backlinks(args.backlinks)
        print(f"\nBacklinks to {args.backlinks}:")
        for link in backlinks:
            print(f"- {link['id']} ({link['date']}): {link['title']}")
    elif args.tag:
        results = indexer.get_by_tag(args.tag)
        print(f"\nNodes tagged '{args.tag}':")
        for _, row in results.iterrows():
            print(f"- {row['id']} ({row['date']}): {row['title']}")
    elif args.orphans:
        orphans = indexer.find_orphans()
        print(f"\nOrphaned nodes ({len(orphans)} total):")
        for orphan in orphans:
            print(f"- {orphan['id']} ({orphan['filename']})")
    elif args.list_tags:
        # Get all nodes and extract unique tags
        all_nodes_df = indexer.nodes_table.to_pandas()
        all_tags = set()

        for _, row in all_nodes_df.iterrows():
            tags = row["tags"]
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()
            if isinstance(tags, list):
                all_tags.update(tags)

        # Count usage for each tag
        tag_counts = {}
        for tag in all_tags:
            count = sum(
                1
                for _, row in all_nodes_df.iterrows()
                if isinstance(row["tags"], (list, np.ndarray)) and tag in row["tags"]
            )
            tag_counts[tag] = count

        # Sort by usage count (most used first), then alphabetically
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))

        print(f"\nAll available tags ({len(sorted_tags)} total):")
        for tag, count in sorted_tags:
            print(f"  {tag} ({count} nodes)")
    elif args.similar_tags:
        # Get all nodes and extract unique tags
        all_nodes_df = indexer.nodes_table.to_pandas()
        all_tags = set()

        for _, row in all_nodes_df.iterrows():
            tags = row["tags"]
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()
            if isinstance(tags, list):
                all_tags.update(tags)

        if args.similar_tags not in all_tags:
            print(f"\nTag '{args.similar_tags}' not found in knowledge graph.")
            print("Use 'make tags' to see available tags.")
        else:
            # Calculate semantic similarity between the query tag and all other tags
            query_embedding = indexer.model.encode(
                args.similar_tags, show_progress_bar=False
            )
            query_norm = query_embedding / np.linalg.norm(query_embedding)

            tag_similarities = []
            for tag in all_tags:
                if tag != args.similar_tags:  # Exclude the query tag itself
                    tag_embedding = indexer.model.encode(tag, show_progress_bar=False)
                    tag_norm = tag_embedding / np.linalg.norm(tag_embedding)
                    similarity = float(query_norm @ tag_norm)

                    # Count usage
                    count = sum(
                        1
                        for _, row in all_nodes_df.iterrows()
                        if isinstance(row["tags"], (list, np.ndarray))
                        and tag in row["tags"]
                    )

                    tag_similarities.append((tag, similarity, count))

            # Sort by similarity (highest first)
            tag_similarities.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTags similar to '{args.similar_tags}' (top {args.limit}):")
            for tag, similarity, count in tag_similarities[: args.limit]:
                print(f"  {tag:<25} (similarity: {similarity:.3f}, {count} nodes)")
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    main()
