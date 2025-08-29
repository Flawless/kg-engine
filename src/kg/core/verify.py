#!/usr/bin/env python3
"""
Knowledge Graph Referential Consistency Checker

Validates that all node: links point to existing nodes, finds duplicate IDs,
identifies orphaned nodes, and validates YAML frontmatter structure.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any

import yaml


class GraphVerifier:
    def __init__(self, kg_path: str):
        self.kg_path = Path(kg_path)
        self.nodes_dir = self.kg_path / "nodes"

        # Data structures for verification
        self.node_ids: Set[str] = set()
        self.node_files: Dict[str, str] = {}  # id -> filename
        self.duplicate_ids: Dict[str, List[str]] = defaultdict(list)
        self.node_links: Set[str] = set()  # All node: links found
        self.all_files: List[str] = []
        self.backlinks: Dict[str, Set[str]] = defaultdict(
            set
        )  # node -> nodes that link to it

        # Error tracking
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown content."""
        if not content.startswith("---\n"):
            return {}, content

        try:
            # Find the closing ---
            end_marker = content.find("\n---\n", 4)
            if end_marker == -1:
                return {}, content

            yaml_content = content[4:end_marker]
            body = content[end_marker + 5 :]

            frontmatter = yaml.safe_load(yaml_content) or {}
            return frontmatter, body
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parse error: {e}")
            return {}, content

    def extract_node_links(self, frontmatter: Dict[str, Any]) -> Set[str]:
        """Extract node: links from frontmatter."""
        links = set()
        if "links" in frontmatter and isinstance(frontmatter["links"], list):
            for link in frontmatter["links"]:
                if isinstance(link, str) and link.startswith("node:"):
                    links.add(link[5:])  # Remove 'node:' prefix
        return links

    def scan_file(self, file_path: Path, is_node: bool = False) -> None:
        """Scan a single file for frontmatter and links."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            self.errors.append(f"Cannot read file {file_path}: encoding error")
            return
        except Exception as e:
            self.errors.append(f"Cannot read file {file_path}: {e}")
            return

        self.all_files.append(str(file_path))
        frontmatter, _ = self.extract_frontmatter(content)

        # Validate frontmatter structure
        if not frontmatter:
            if is_node:
                self.errors.append(
                    f"Node {file_path.name}: Missing or invalid frontmatter"
                )
            return

        # Check for node ID (required for nodes)
        if is_node:
            if "id" not in frontmatter:
                self.errors.append(f"Node {file_path.name}: Missing 'id' field")
                return

            node_id = frontmatter["id"]
            if not isinstance(node_id, str):
                self.errors.append(f"Node {file_path.name}: 'id' must be a string")
                return

            # Track node IDs and duplicates
            if node_id in self.node_ids:
                self.duplicate_ids[node_id].append(file_path.name)
            else:
                self.node_ids.add(node_id)
                self.node_files[node_id] = file_path.name
                self.duplicate_ids[node_id].append(file_path.name)

            # Extract node links and track backlinks
            node_links = self.extract_node_links(frontmatter)
            self.node_links.update(node_links)

            # Track who links to whom
            for linked_node in node_links:
                self.backlinks[linked_node].add(node_id)

    def scan_directory(self) -> None:
        """Scan all markdown files in nodes/ directory."""
        if self.nodes_dir.exists():
            for file_path in self.nodes_dir.glob("*.md"):
                self.scan_file(file_path, is_node=True)
        else:
            self.errors.append(f"Nodes directory not found: {self.nodes_dir}")

    def check_broken_links(self) -> None:
        """Check for node: links that point to non-existent nodes."""
        broken_links = self.node_links - self.node_ids
        for broken_link in sorted(broken_links):
            self.errors.append(
                f"Broken link: node:{broken_link} (target does not exist)"
            )

    def check_duplicate_ids(self) -> None:
        """Check for duplicate node IDs."""
        for node_id, files in self.duplicate_ids.items():
            if len(files) > 1:
                file_list = ", ".join(files)
                self.errors.append(
                    f"Duplicate node ID '{node_id}' in files: {file_list}"
                )

    def find_orphaned_nodes(self) -> None:
        """Find nodes that are not referenced by any node: links."""
        referenced_nodes = self.node_links
        orphaned = self.node_ids - referenced_nodes

        for orphaned_node in sorted(orphaned):
            filename = self.node_files[orphaned_node]
            self.warnings.append(f"Orphaned node: {orphaned_node} ({filename})")

    def verify(self) -> bool:
        """Run all verification checks."""
        print(f"Scanning knowledge graph at: {self.kg_path}")
        print()

        # Scan all files
        self.scan_directory()

        # Run checks
        self.check_duplicate_ids()
        self.check_broken_links()
        self.find_orphaned_nodes()

        # Report results
        self.print_report()

        return len(self.errors) == 0

    def print_report(self) -> None:
        """Print verification report."""
        print(f"Scanned {len(self.all_files)} files")
        print(f"Found {len(self.node_ids)} unique nodes")
        print(f"Found {len(self.node_links)} node references")
        print()

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ❌ {error}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ No issues found - knowledge graph is consistent")
        elif not self.errors:
            print("✅ No critical errors - knowledge graph structure is valid")
        else:
            print(f"❌ Found {len(self.errors)} errors that need to be fixed")


def main(kg_path=None):
    """Main entry point."""
    if kg_path is None:
        # Default to current working directory
        kg_path = os.getcwd()
    verifier = GraphVerifier(kg_path)
    success = verifier.verify()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
