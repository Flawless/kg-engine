#!/usr/bin/env python3
"""
Knowledge Graph Traversal

Provides graph traversal capabilities for exploring node relationships.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import deque, defaultdict

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value


class GraphTraverser:
    def __init__(self, kg_path: str = "."):
        self.kg_path = Path(kg_path)
        self.nodes = {}  # id -> node data
        self.edges = defaultdict(set)  # source -> set of targets
        self.reverse_edges = defaultdict(set)  # target -> set of sources
        self.load_graph()

    def load_graph(self):
        """Load the entire graph into memory."""
        nodes_dir = self.kg_path / "nodes"
        if not nodes_dir.exists():
            print(f"Nodes directory not found: {nodes_dir}")
            return

        # First pass: load all nodes
        for file_path in nodes_dir.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            frontmatter, title = self.extract_frontmatter(content)
            if "id" not in frontmatter:
                continue

            node_id = frontmatter["id"]
            self.nodes[node_id] = {
                "id": node_id,
                "filename": file_path.name,
                "title": title,
                "date": frontmatter.get("date", ""),
                "tags": frontmatter.get("tags", []),
                "links": frontmatter.get("links", []),
            }

        # Second pass: build edges
        for node_id, node_data in self.nodes.items():
            for link in node_data["links"]:
                if isinstance(link, str):
                    if link.startswith("node:"):
                        target_id = link[5:]
                        if target_id in self.nodes:  # Only add edges to existing nodes
                            self.edges[node_id].add(target_id)
                            self.reverse_edges[target_id].add(node_id)
                    elif ":" in link:
                        # Handle other link types like sprint:, project:, task:
                        # Create virtual nodes for these references
                        self.edges[node_id].add(link)
                        self.reverse_edges[link].add(node_id)

    def extract_frontmatter(self, content: str) -> Tuple[Dict, str]:
        """Extract YAML frontmatter and title."""
        if not content.startswith("---\n"):
            return {}, ""

        try:
            end_marker = content.find("\n---\n", 4)
            if end_marker == -1:
                return {}, ""

            yaml_content = content[4:end_marker]
            body = content[end_marker + 5 :]

            # Extract title
            title = ""
            for line in body.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            frontmatter = yaml.safe_load(yaml_content) or {}
            return frontmatter, title
        except yaml.YAMLError:
            return {}, ""

    def bfs_traverse(
        self, start_id: str, max_depth: int = 3, direction: str = "both"
    ) -> Dict[int, List[str]]:
        """Breadth-first traversal from a starting node."""
        # Check if it's a virtual node (like sprint:2025-08-04)
        if (
            start_id not in self.nodes
            and start_id not in self.reverse_edges
            and start_id not in self.edges
        ):
            print(f"Node {start_id} not found")
            return {}

        visited = {start_id}
        levels = {0: [start_id]}
        queue = deque([(start_id, 0)])

        while queue and max(levels.keys()) < max_depth:
            current_id, depth = queue.popleft()

            if depth < max_depth:
                # Get neighbors based on direction
                neighbors = set()
                if direction in ["out", "both"]:
                    neighbors.update(self.edges.get(current_id, set()))
                if direction in ["in", "both"]:
                    neighbors.update(self.reverse_edges.get(current_id, set()))

                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

                        if depth + 1 not in levels:
                            levels[depth + 1] = []
                        levels[depth + 1].append(neighbor_id)

        return levels

    def find_path(
        self, start_id: str, end_id: str, max_length: int = 6
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        queue = deque([(start_id, [start_id])])

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_length:
                continue

            for neighbor_id in self.edges.get(current_id, set()):
                if neighbor_id == end_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def find_clusters(self, min_size: int = 3) -> List[Set[str]]:
        """Find connected components (clusters) in the graph."""
        visited = set()
        clusters = []

        for node_id in self.nodes:
            if node_id not in visited:
                # BFS to find all connected nodes
                cluster = set()
                queue = deque([node_id])

                while queue:
                    current_id = queue.popleft()
                    if current_id in visited:
                        continue

                    visited.add(current_id)
                    cluster.add(current_id)

                    # Add both incoming and outgoing connections
                    for neighbor_id in self.edges.get(current_id, set()):
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)

                    for neighbor_id in self.reverse_edges.get(current_id, set()):
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)

                if len(cluster) >= min_size:
                    clusters.append(cluster)

        return sorted(clusters, key=len, reverse=True)

    def get_node_centrality(self) -> List[Tuple[str, int, int, int]]:
        """Calculate node centrality metrics."""
        centrality = []

        for node_id in self.nodes:
            in_degree = len(self.reverse_edges.get(node_id, set()))
            out_degree = len(self.edges.get(node_id, set()))
            total_degree = in_degree + out_degree

            centrality.append((node_id, total_degree, in_degree, out_degree))

        return sorted(centrality, key=lambda x: x[1], reverse=True)

    def export_graphviz(
        self, output_file: str = "graph.dot", highlight_nodes: Set[str] = None
    ):
        """Export graph in Graphviz DOT format."""
        with open(output_file, "w") as f:
            f.write("digraph KnowledgeGraph {\n")
            f.write("  rankdir=LR;\n")
            f.write("  node [shape=box, style=rounded];\n")

            # Write nodes
            for node_id, node_data in self.nodes.items():
                label = f"{node_id}\\n{node_data['title'][:30]}..."
                color = "lightblue"

                if highlight_nodes and node_id in highlight_nodes:
                    color = "yellow"

                f.write(
                    f'  "{node_id}" [label="{label}", fillcolor={color}, style=filled];\n'
                )

            # Write edges
            for source_id, targets in self.edges.items():
                for target_id in targets:
                    f.write(f'  "{source_id}" -> "{target_id}";\n')

            f.write("}\n")

        print(f"Graph exported to {output_file}")
        print(f"Visualize with: dot -Tpng {output_file} -o graph.png")


def print_traversal_tree(traversal: Dict[int, List[str]], nodes: Dict[str, Dict]):
    """Pretty print traversal results as a tree."""
    for depth, node_ids in sorted(traversal.items()):
        for node_id in node_ids:
            indent = "  " * depth
            if node_id in nodes:
                node = nodes.get(node_id, {})
                title = node.get("title", "Unknown")[:50]
                print(f"{indent}{'└─' if depth > 0 else ''} {node_id}: {title}")
            else:
                # Virtual node (sprint:, project:, task:, etc.)
                print(f"{indent}{'└─' if depth > 0 else ''} {node_id}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Traversal")
    parser.add_argument("--traverse", help="Traverse from node ID")
    parser.add_argument("--depth", type=int, default=3, help="Max traversal depth")
    parser.add_argument(
        "--direction",
        choices=["in", "out", "both"],
        default="both",
        help="Traversal direction",
    )
    parser.add_argument(
        "--path", nargs=2, metavar=("START", "END"), help="Find path between two nodes"
    )
    parser.add_argument("--clusters", action="store_true", help="Find node clusters")
    parser.add_argument(
        "--centrality", action="store_true", help="Show node centrality"
    )
    parser.add_argument("--export", help="Export to Graphviz DOT file")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")

    args = parser.parse_args()
    traverser = GraphTraverser()

    if args.traverse:
        print(
            f"\nTraversing from '{args.traverse}' (depth={args.depth}, direction={args.direction}):\n"
        )
        traversal = traverser.bfs_traverse(args.traverse, args.depth, args.direction)
        print_traversal_tree(traversal, traverser.nodes)

        # Summary
        total_nodes = sum(len(nodes) for nodes in traversal.values())
        print(f"\nFound {total_nodes} nodes within {args.depth} steps")

    elif args.path:
        start, end = args.path
        path = traverser.find_path(start, end)
        if path:
            print(f"\nPath from '{start}' to '{end}':")
            for i, node_id in enumerate(path):
                node = traverser.nodes.get(node_id, {})
                print(f"  {i + 1}. {node_id}: {node.get('title', 'Unknown')}")
        else:
            print(f"\nNo path found from '{start}' to '{end}'")

    elif args.clusters:
        clusters = traverser.find_clusters()
        print(f"\nFound {len(clusters)} clusters:\n")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i + 1} ({len(cluster)} nodes):")
            for node_id in sorted(cluster)[:10]:  # Show first 10
                node = traverser.nodes.get(node_id, {})
                print(f"  - {node_id}: {node.get('title', 'Unknown')}")
            if len(cluster) > 10:
                print(f"  ... and {len(cluster) - 10} more")
            print()

    elif args.centrality:
        centrality = traverser.get_node_centrality()
        print("\nNode Centrality (total, in, out):\n")
        for node_id, total, in_deg, out_deg in centrality[:20]:
            node = traverser.nodes.get(node_id, {})
            print(
                f"{node_id:30} {total:3} ({in_deg:2} in, {out_deg:2} out) - {node.get('title', '')[:40]}"
            )

    elif args.export:
        # If traversing, highlight traversed nodes
        highlight = set()
        if args.traverse:
            traversal = traverser.bfs_traverse(
                args.traverse, args.depth, args.direction
            )
            highlight = set(
                node_id for nodes in traversal.values() for node_id in nodes
            )

        traverser.export_graphviz(args.export, highlight)

    elif args.stats:
        total_nodes = len(traverser.nodes)
        total_edges = sum(len(targets) for targets in traverser.edges.values())
        orphans = [n for n in traverser.nodes if n not in traverser.reverse_edges]
        leaves = [n for n in traverser.nodes if n not in traverser.edges]

        print("\nGraph Statistics:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {total_edges}")
        print(f"  Orphaned nodes: {len(orphans)}")
        print(f"  Leaf nodes: {len(leaves)}")
        print(f"  Average degree: {(2 * total_edges) / total_nodes:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
