"""
investigation/subgraph.py
--------------------------
Subgraph expansion and statistics for the AML investigation layer.
"""

import networkx as nx
import os
from typing import Dict, Any

# Hard cap: subgraphs larger than this are trimmed to avoid
# slow cycle-detection on very dense ego-networks.
MAX_SUBGRAPH_NODES = 500


def expand_subgraph(G: nx.DiGraph, node_id: str, k: int = 2) -> nx.DiGraph:
    """
    Perform a k-hop BFS from *node_id* and return a copy of the induced subgraph.

    Parameters
    ----------
    G       : full transaction DiGraph
    node_id : alert node to expand from
    k       : number of hops (default 2)

    Returns
    -------
    nx.DiGraph — a *copy* of the k-hop induced subgraph (capped at MAX_SUBGRAPH_NODES).
    """
    if node_id not in G:
        return nx.DiGraph()

    # BFS using the undirected view so we capture both up- and down-stream hops.
    G_undirected = G.to_undirected()
    
    # Manual BFS
    visited = {node_id}
    queue = [(node_id, 0)]
    
    is_inference = os.environ.get("AML_INFERENCE", "0") == "1"
    
    if is_inference:
        MAX_NODES = 40
        MAX_NEIGHBORS = 8
        
        while queue and len(visited) < MAX_NODES:
            curr, depth = queue.pop(0)
            if depth < k:
                neighbors = list(G_undirected.neighbors(curr))[:MAX_NEIGHBORS]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        if len(visited) >= MAX_NODES:
                            break
    else:
        while queue and len(visited) < MAX_SUBGRAPH_NODES:
            curr, depth = queue.pop(0)
            if depth < k:
                for neighbor in G_undirected.neighbors(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        if len(visited) >= MAX_SUBGRAPH_NODES:
                            break

    return G.subgraph(visited).copy()


def subgraph_stats(G_sub: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute basic structural statistics for a subgraph.

    Returns
    -------
    dict with keys: num_nodes, num_edges, density
    """
    num_nodes = G_sub.number_of_nodes()
    num_edges = G_sub.number_of_edges()
    density = nx.density(G_sub) if num_nodes > 1 else 0.0
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
    }
