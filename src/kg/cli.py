#!/usr/bin/env python3
"""KG CLI - unified command-line interface for knowledge graph operations."""

import click
import sys
import os
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent))

from core import index, graph, verify


@click.group()
@click.pass_context
def cli(ctx):
    """KG - Knowledge Graph with semantic search."""
    # Ensure we're in the right directory (where nodes/ exists)
    if not Path("nodes").exists():
        # Try parent directory
        parent_nodes = Path("../nodes")
        if parent_nodes.exists():
            os.chdir("..")
    
    ctx.ensure_object(dict)


@cli.command()
@click.option('--rebuild', is_flag=True, help='Rebuild entire index')
@click.option('--update', help='Update single file in index')
@click.option('--stats', is_flag=True, help='Show index statistics')
def index_cmd(rebuild, update, stats):
    """Manage the search index."""
    # Call the original index-graph.py main with appropriate arguments
    sys.argv = ['index-graph.py']
    if rebuild:
        sys.argv.append('--rebuild')
    if update:
        sys.argv.extend(['--update', update])
    if stats:
        sys.argv.append('--stats')
    
    # Import and run the main function from index module
    if hasattr(index, 'main'):
        index.main()
    else:
        # Fallback: execute the module directly
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Number of results')
@click.option('--content', is_flag=True, help='Search in content too')
@click.option('--verbose', is_flag=True, help='Show detailed results')
def search(query, limit, content, verbose):
    """Search the knowledge graph."""
    sys.argv = ['index-graph.py', '--search', query, '--limit', str(limit)]
    if content:
        sys.argv.append('--content')
    if verbose:
        sys.argv.append('--verbose')
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
@click.argument('node_id')
@click.option('--depth', default=3, help='Traversal depth')
@click.option('--direction', default='both', type=click.Choice(['both', 'in', 'out']))
def traverse(node_id, depth, direction):
    """Traverse graph from a node."""
    sys.argv = ['graph-traverse.py', '--traverse', node_id, '--depth', str(depth), '--direction', direction]
    
    if hasattr(graph, 'main'):
        graph.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'graph.py').read())


@cli.command()
@click.argument('tag_name')
def tag(tag_name):
    """Find nodes by tag."""
    sys.argv = ['index-graph.py', '--tag', tag_name]
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
def tags():
    """List all tags with counts."""
    sys.argv = ['index-graph.py', '--list-tags']
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
@click.argument('node_id')
@click.option('--limit', default=5, help='Number of similar nodes')
def similar(node_id, limit):
    """Find similar nodes."""
    sys.argv = ['index-graph.py', '--similar', node_id, '--limit', str(limit)]
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
@click.argument('node_id')
def backlinks(node_id):
    """Show what links to a node."""
    sys.argv = ['index-graph.py', '--backlinks', node_id]
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
def orphans():
    """Find orphaned nodes."""
    sys.argv = ['index-graph.py', '--orphans']
    
    if hasattr(index, 'main'):
        index.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'index.py').read())


@cli.command()
def verify_cmd():
    """Verify graph consistency."""
    sys.argv = ['verify-graph.py']
    
    if hasattr(verify, 'main'):
        verify.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'verify.py').read())


@cli.command()
def centrality():
    """Show most connected nodes."""
    sys.argv = ['graph-traverse.py', '--centrality']
    
    if hasattr(graph, 'main'):
        graph.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'graph.py').read())


@cli.command()
def clusters():
    """Find node clusters."""
    sys.argv = ['graph-traverse.py', '--clusters']
    
    if hasattr(graph, 'main'):
        graph.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'graph.py').read())


@cli.command()
@click.argument('from_node')
@click.argument('to_node')
def path(from_node, to_node):
    """Find path between two nodes."""
    sys.argv = ['graph-traverse.py', '--path', from_node, to_node]
    
    if hasattr(graph, 'main'):
        graph.main()
    else:
        exec(open(Path(__file__).parent / 'core' / 'graph.py').read())


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()