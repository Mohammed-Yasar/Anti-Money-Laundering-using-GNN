"""
Network generation module.
Creates structural affinity graph using Barabási–Albert model.
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Tuple, Dict
from . import config


def create_structural_graph(account_ids: list, seed: int = None) -> nx.DiGraph:
    """
    Create a directed graph representing structural affinity between accounts.
    Uses Barabási–Albert model for heavy-tailed degree distribution.
    
    Args:
        account_ids: List of account IDs
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX DiGraph with account IDs as nodes
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_nodes = len(account_ids)
    
    # Generate undirected BA graph
    ba_graph = nx.barabasi_albert_graph(
        n=num_nodes,
        m=config.BARABASI_ALBERT_M,
        seed=seed
    )
    
    # Convert to directed graph by randomly assigning edge directions
    digraph = nx.DiGraph()
    
    for edge in ba_graph.edges():
        # Randomly decide direction
        if np.random.random() < 0.5:
            digraph.add_edge(edge[0], edge[1])
        else:
            digraph.add_edge(edge[1], edge[0])
    
    # Add some additional random edges for realism
    num_random_edges = int(num_nodes * 0.1)
    for _ in range(num_random_edges):
        source = np.random.randint(0, num_nodes)
        target = np.random.randint(0, num_nodes)
        if source != target:
            digraph.add_edge(source, target)
    
    # Map node indices to account IDs
    mapping = {i: account_ids[i] for i in range(num_nodes)}
    digraph = nx.relabel_nodes(digraph, mapping)
    
    return digraph


def compute_degree_statistics(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute degree distribution statistics.
    
    Args:
        graph: NetworkX DiGraph
        
    Returns:
        Dictionary with degree statistics
    """
    degrees = [d for n, d in graph.degree()]
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    
    stats = {
        'mean_degree': np.mean(degrees),
        'median_degree': np.median(degrees),
        'max_degree': np.max(degrees),
        'min_degree': np.min(degrees),
        'std_degree': np.std(degrees),
        'mean_in_degree': np.mean(in_degrees),
        'mean_out_degree': np.mean(out_degrees),
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges()
    }
    
    return stats


def get_neighbors(graph: nx.DiGraph, node: str) -> list:
    """
    Get outgoing neighbors of a node (accounts this node can send to).
    
    Args:
        graph: NetworkX DiGraph
        node: Node ID
        
    Returns:
        List of neighbor node IDs
    """
    if node not in graph:
        return []
    return list(graph.successors(node))


def save_degree_statistics(stats: Dict[str, float], output_path: str) -> None:
    """
    Save degree statistics to file.
    
    Args:
        stats: Dictionary of statistics
        output_path: Path to save statistics
    """
    with open(output_path, 'w') as f:
        f.write("Network Degree Statistics\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
    
    print(f"Saved degree statistics to {output_path}")
