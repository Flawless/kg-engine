#!/usr/bin/env python3
"""KG CLI - unified command-line interface for knowledge graph operations."""

import click
import sys
import os
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent))

from core import index as index_module, graph as graph_module, verify as verify_module


def get_kg_dir():
    """Get the knowledge graph directory."""
    # Check current directory
    if Path("nodes").exists():
        return Path.cwd()

    # Check parent directory
    parent = Path.cwd().parent
    if (parent / "nodes").exists():
        return parent

    # Check environment variable
    if kg_dir := os.environ.get("KG_DIR"):
        kg_path = Path(kg_dir)
        if (kg_path / "nodes").exists():
            return kg_path

    # Default to current directory
    return Path.cwd()


@click.group()
@click.option("--config", help="Path to configuration file")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, debug):
    """KG - Knowledge Graph with semantic search."""
    # Set up logging
    import logging

    if debug:
        logging.basicConfig(
            level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s"
        )
    else:
        # Default to WARNING level to suppress noise
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        # Suppress sentence-transformers and other noisy libraries
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

        # Also set transformers library to error mode to suppress all progress bars
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()

        # Disable tqdm globally
        import os

        os.environ["TQDM_DISABLE"] = "1"

    # Store KG directory in context
    ctx.ensure_object(dict)
    ctx.obj["kg_dir"] = str(get_kg_dir())
    ctx.obj["config"] = config
    ctx.obj["debug"] = debug


@cli.command()
@click.option("--rebuild", is_flag=True, help="Rebuild entire index")
@click.option("--update", help="Update single file in index")
@click.option("--stats", is_flag=True, help="Show index statistics")
@click.pass_context
def index(ctx, rebuild, update, stats):
    """Manage the search index."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    # Call the original index-graph.py main with appropriate arguments
    sys.argv = ["index-graph.py"]
    if rebuild:
        sys.argv.append("--rebuild")
    if update:
        sys.argv.extend(["--update", update])
    if stats:
        sys.argv.append("--stats")

    # Set environment variables for the indexer
    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    # Import and run the main function from index module
    index_module.main()


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, help="Number of results")
@click.option("--content", is_flag=True, help="Search in content too")
@click.option("--verbose", is_flag=True, help="Show detailed results")
@click.pass_context
def search(ctx, query, limit, content, verbose):
    """Search the knowledge graph."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--search", query, "--limit", str(limit)]
    if content:
        sys.argv.append("--content")
    if verbose:
        sys.argv.append("--verbose")

    index_module.main()


@cli.command()
@click.argument("node_id")
@click.option("--depth", default=3, help="Traversal depth")
@click.option("--direction", default="both", type=click.Choice(["both", "in", "out"]))
@click.pass_context
def traverse(ctx, node_id, depth, direction):
    """Traverse graph from a node."""
    kg_dir = ctx.obj["kg_dir"]

    sys.argv = [
        "graph-traverse.py",
        "--traverse",
        node_id,
        "--depth",
        str(depth),
        "--direction",
        direction,
    ]

    graph_module.main(kg_dir)


@cli.command()
@click.argument("tag_name")
@click.pass_context
def tag(ctx, tag_name):
    """Find nodes by tag."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--tag", tag_name]
    index_module.main()


@cli.command()
@click.pass_context
def tags(ctx):
    """List all tags with counts."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--list-tags"]
    index_module.main()


@cli.command()
@click.argument("node_id")
@click.option("--limit", default=5, help="Number of similar nodes")
@click.pass_context
def similar(ctx, node_id, limit):
    """Find similar nodes."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--similar", node_id, "--limit", str(limit)]
    index_module.main()


@cli.command()
@click.argument("node_id")
@click.pass_context
def backlinks(ctx, node_id):
    """Show what links to a node."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--backlinks", node_id]
    index_module.main()


@cli.command()
@click.pass_context
def orphans(ctx):
    """Find orphaned nodes."""
    kg_dir = ctx.obj["kg_dir"]
    config = ctx.obj.get("config")

    os.environ["KG_PATH"] = kg_dir
    if config:
        os.environ["KG_CONFIG"] = config

    sys.argv = ["index-graph.py", "--orphans"]
    index_module.main()


@cli.command()
@click.pass_context
def verify(ctx):
    """Verify graph consistency."""
    kg_dir = ctx.obj["kg_dir"]
    sys.argv = ["verify-graph.py"]

    verify_module.main(kg_dir)


@cli.command()
@click.pass_context
def centrality(ctx):
    """Show most connected nodes."""
    kg_dir = ctx.obj["kg_dir"]

    sys.argv = ["graph-traverse.py", "--centrality"]
    graph_module.main(kg_dir)


@cli.command()
@click.pass_context
def clusters(ctx):
    """Find node clusters."""
    kg_dir = ctx.obj["kg_dir"]

    sys.argv = ["graph-traverse.py", "--clusters"]
    graph_module.main(kg_dir)


@cli.command()
@click.argument("from_node")
@click.argument("to_node")
@click.pass_context
def path(ctx, from_node, to_node):
    """Find path between two nodes."""
    kg_dir = ctx.obj["kg_dir"]

    sys.argv = ["graph-traverse.py", "--path", from_node, to_node]
    graph_module.main(kg_dir)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
