"""
Plotting utilities for visualization and validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional


def plot_degree_distribution_loglog(
    graph: nx.DiGraph,
    output_path: str,
    title: str = "Network Degree Distribution (Log-Log)"
) -> None:
    """
    Plot degree distribution on log-log scale to verify heavy tail.
    
    Args:
        graph: NetworkX DiGraph
        output_path: Path to save plot
        title: Plot title
    """
    # Get degree sequence
    degrees = [d for n, d in graph.degree()]
    
    # Compute degree distribution
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(degree_counts.index, degree_counts.values, 'bo-', alpha=0.7)
    plt.xlabel('Degree (k)', fontsize=12)
    plt.ylabel('Frequency P(k)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add power-law reference line
    x = np.array(degree_counts.index)
    y_ref = x[0]**2 / x**2 * degree_counts.values[0]
    plt.loglog(x, y_ref, 'r--', alpha=0.5, label='Power-law reference')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved degree distribution plot to {output_path}")


def plot_daily_transaction_volume(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Daily Transaction Volume"
) -> None:
    """
    Plot daily transaction volume over simulation period.
    
    Args:
        df: DataFrame with transactions
        output_path: Path to save plot
        title: Plot title
    """
    # Extract date from timestamp
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
    
    # Aggregate by date
    daily_stats = df_copy.groupby('date').agg({
        'transaction_id': 'count',
        'amount': 'sum',
        'laundering_flag': 'sum'
    }).rename(columns={
        'transaction_id': 'total_txs',
        'amount': 'total_volume',
        'laundering_flag': 'laundering_txs'
    })
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Transaction count
    axes[0].plot(daily_stats.index, daily_stats['total_txs'], 'b-', linewidth=1.5, alpha=0.7, label='Total')
    axes[0].plot(daily_stats.index, daily_stats['laundering_txs'], 'r-', linewidth=2.0, label='Laundering', marker='o', markersize=3)
    axes[0].fill_between(daily_stats.index, 0, daily_stats['laundering_txs'], alpha=0.3, color='red')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Number of Transactions', fontsize=11)
    axes[0].set_title('Daily Transaction Count', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Transaction volume
    axes[1].plot(daily_stats.index, daily_stats['total_volume'], 'g-', linewidth=1.5)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Total Volume ($)', fontsize=11)
    axes[1].set_title('Daily Transaction Volume', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved daily volume plot to {output_path}")


def plot_amount_distribution(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Transaction Amount Distribution"
) -> None:
    """
    Plot transaction amount distribution (legitimate vs laundering).
    
    Args:
        df: DataFrame with transactions
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate data
    legitimate = df[df['laundering_flag'] == 0]['amount']
    laundering = df[df['laundering_flag'] == 1]['amount']
    
    # Histogram
    axes[0].hist(legitimate, bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[0].hist(laundering, bins=50, alpha=0.7, label='Laundering', color='red', density=True)
    axes[0].set_xlabel('Amount ($)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Amount Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log-scale histogram
    axes[1].hist(np.log10(legitimate + 1), bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[1].hist(np.log10(laundering + 1), bins=50, alpha=0.7, label='Laundering', color='red', density=True)
    axes[1].set_xlabel('log10(Amount + 1)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Amount Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved amount distribution plot to {output_path}")


def plot_network_overview(
    graph: nx.DiGraph,
    output_path: str,
    title: str = "Network Structure Overview",
    max_nodes: int = 500
) -> None:
    """
    Plot network structure visualization (sampled for large graphs).
    
    Args:
        graph: NetworkX DiGraph
        output_path: Path to save plot
        title: Plot title
        max_nodes: Maximum nodes to visualize
    """
    # Sample subgraph if too large
    if graph.number_of_nodes() > max_nodes:
        # Select high-degree nodes
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_plot = [n for n, d in top_nodes]
        subgraph = graph.subgraph(nodes_to_plot)
    else:
        subgraph = graph
    
    # Plot
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    
    # Node sizes based on degree
    degrees = dict(subgraph.degree())
    node_sizes = [max(10, degrees[n] * 5) for n in subgraph.nodes()]
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, arrows=True, arrowsize=5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network overview plot to {output_path}")


def plot_campaign_statistics(
    campaign_metadata: list,
    output_path: str,
    title: str = "Campaign Statistics"
) -> None:
    """
    Plot campaign-level statistics showing transaction counts and typologies.
    
    Args:
        campaign_metadata: List of campaign metadata dictionaries
        output_path: Path to save plot
        title: Plot title
    """
    if not campaign_metadata:
        print("No campaign metadata available for plotting")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Extract data
    campaign_ids = [c['campaign_id'] for c in campaign_metadata]
    tx_counts = [c['num_transactions'] for c in campaign_metadata]
    typologies = [c['typology'] for c in campaign_metadata]
    
    # Color mapping
    color_map = {'smurfing': 'red', 'layering': 'orange', 'circular': 'purple'}
    colors = [color_map.get(t, 'gray') for t in typologies]
    
    # Plot 1: Transaction count per campaign
    axes[0].bar(campaign_ids, tx_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Campaign ID', fontsize=11)
    axes[0].set_ylabel('Number of Transactions', fontsize=11)
    axes[0].set_title('Transactions per Campaign', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=np.mean(tx_counts), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(tx_counts):.0f}')
    axes[0].legend()
    
    # Plot 2: Typology distribution
    typology_counts = pd.Series(typologies).value_counts()
    axes[1].bar(typology_counts.index, typology_counts.values, color=[color_map[t] for t in typology_counts.index], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Typology', fontsize=11)
    axes[1].set_ylabel('Number of Campaigns', fontsize=11)
    axes[1].set_title('Campaign Typology Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (typ, count) in enumerate(typology_counts.items()):
        axes[1].text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved campaign statistics plot to {output_path}")

