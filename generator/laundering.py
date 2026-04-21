"""
Money laundering pattern injection module.
Implements three typologies: Smurfing, Layering, and Circular.
UPDATED: Heavier campaigns with temporal compression for 2-3% laundering ratio.
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from . import config


def inject_laundering_campaigns(
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    seed: int = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Inject money laundering campaigns with three typologies.
    Target: 15-25% node reuse across campaigns with randomized roles.
    """
    if seed is not None:
        np.random.seed(seed)
    
    laundering_nodes = _select_laundering_nodes(accounts_df, graph)
    laundering_nodes_list = list(laundering_nodes)
    
    all_laundering_txs = []
    campaign_metadata = []
    
    num_campaigns = config.LAUNDERING_CAMPAIGNS
    smurfing_count = int(num_campaigns * 0.40)
    layering_count = int(num_campaigns * 0.35)
    circular_count = num_campaigns - smurfing_count - layering_count
    
    def distribute_splits(total: int) -> list[str]:
        train_n = int(total * 0.6)
        val_n = int(total * 0.2)
        test_n = total - train_n - val_n
        return ['train'] * train_n + ['val'] * val_n + ['test'] * test_n

    campaign_assignments = (
        [('smurfing', s) for s in distribute_splits(smurfing_count)] +
        [('layering', s) for s in distribute_splits(layering_count)] +
        [('circular', s) for s in distribute_splits(circular_count)]
    )
    np.random.shuffle(campaign_assignments)
    
    start_sim = datetime(2024, 1, 1)
    
    # Track unique nodes used vs total slots to maintain REUSE_RATE_CAP
    unique_nodes_used = set()
    total_slots_filled = 0
    
    def sample_start_date(split_name: str) -> datetime:
        if split_name == 'train':
            offset_days = np.random.randint(0, 56)
        elif split_name == 'val':
            offset_days = np.random.randint(60, 71)
        else:
            offset_days = np.random.randint(75, 86)
        return start_sim + timedelta(days=float(offset_days), hours=float(np.random.randint(0, 24)))
    
    campaign_id = 1
    for typology, split_name in campaign_assignments:
        start_date = sample_start_date(split_name)
        
        # Estimate slots for this campaign (roughly 25-35 nodes)
        # We need (Total Slots - Unique Nodes) / Total Slots <= REUSE_RATE_CAP
        # Which implies Unique Nodes >= Total Slots * (1 - REUSE_RATE_CAP)
        
        needed = 35 
        current_reuse_rate = (total_slots_filled - len(unique_nodes_used)) / total_slots_filled if total_slots_filled > 0 else 0
        
        # Force fresh nodes if we are over the reuse cap
        if current_reuse_rate >= config.REUSE_RATE_CAP:
            reuse_count = 0
        else:
            # Allow some reuse but stay cautious
            max_reuse_possible = int(total_slots_filled * config.REUSE_RATE_CAP) - (total_slots_filled - len(unique_nodes_used))
            reuse_count = min(int(needed * 0.25), max(0, max_reuse_possible))
        
        reused = []
        if reuse_count > 0 and len(unique_nodes_used) > 10:
            reused = np.random.choice(list(unique_nodes_used), size=min(reuse_count, len(unique_nodes_used)), replace=False).tolist()
            
        remaining_needed = needed - len(reused)
        fresh_pool = [n for n in laundering_nodes_list if n not in unique_nodes_used]
        
        if len(fresh_pool) < remaining_needed:
            # Emergency: if we run out of fresh nodes, we MUST reuse, but we warned the user about REUSE_RATE_CAP
            fresh = fresh_pool
        else:
            fresh = np.random.choice(fresh_pool, size=remaining_needed, replace=False).tolist()
            
        campaign_pool = set(reused + fresh)
        
        # Update tracking
        unique_nodes_used.update(campaign_pool)
        total_slots_filled += len(campaign_pool)
        
        if typology == 'smurfing':
            txs, metadata = _generate_smurfing_campaign(
                campaign_id=campaign_id, accounts_df=accounts_df, graph=graph,
                laundering_nodes=campaign_pool, start_date=start_date
            )
        elif typology == 'layering':
            txs, metadata = _generate_layering_campaign(
                campaign_id=campaign_id, accounts_df=accounts_df, graph=graph,
                laundering_nodes=campaign_pool, start_date=start_date
            )
        else:
            txs, metadata = _generate_circular_campaign(
                campaign_id=campaign_id, accounts_df=accounts_df, graph=graph,
                laundering_nodes=campaign_pool, start_date=start_date
            )
            
        all_laundering_txs.extend(txs)
        campaign_metadata.append(metadata)
        campaign_id += 1
    
    return all_laundering_txs, campaign_metadata


def _get_degree_bins(graph: nx.DiGraph) -> Dict[str, str]:
    """
    Divide nodes into High, Mid, and Low degree bins once globally.
    High: top 15%, Mid: next 50%, Low: bottom 35%.
    """
    degrees = dict(graph.degree())
    nodes_sorted = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    n = len(nodes_sorted)
    
    high_cutoff = int(n * 0.15)
    mid_cutoff = int(n * 0.65)
    
    bins = {}
    for i, node in enumerate(nodes_sorted):
        if i < high_cutoff:
            bins[node] = "high"
        elif i < mid_cutoff:
            bins[node] = "mid"
        else:
            bins[node] = "low"
    return bins


def _select_laundering_nodes(
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph
) -> Set[str]:
    """
    Select accounts for laundering pool using stratified degree sampling.
    Target: 30% High, 40% Mid, 30% Low.
    """
    num_laundering = int(len(accounts_df) * config.LAUNDERING_NODE_RATIO)
    degree_bins = _get_degree_bins(graph)
    
    nodes_by_bin = {"high": [], "mid": [], "low": []}
    for acc in accounts_df['account_id']:
        b = degree_bins.get(acc, "low")
        nodes_by_bin[b].append(acc)
        
    counts = {
        "high": int(num_laundering * 0.30),
        "mid": int(num_laundering * 0.40),
        "low": num_laundering - int(num_laundering * 0.30) - int(num_laundering * 0.40)
    }
    
    selected = set()
    for b, count in counts.items():
        available = nodes_by_bin[b]
        if not available:
            continue
        # Use replacement=False to ensure unique participants
        chosen = np.random.choice(available, size=min(count, len(available)), replace=False)
        selected.update(chosen)
    
    return selected


def _get_legit_intermediary(
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    exclude_nodes: Set[str]
) -> str:
    """
    Select a legitimate intermediary node.
    Sampling: 70% mid-degree, 30% low-degree.
    Never samples nodes in exclude_nodes (current campaign participants).
    """
    degree_bins = _get_degree_bins(graph)
    
    # Filter nodes by bin and exclude ALL current campaign participants
    mid_nodes = [n for n, b in degree_bins.items() if b == 'mid' and n not in exclude_nodes]
    low_nodes = [n for n, b in degree_bins.items() if b == 'low' and n not in exclude_nodes]
    
    # Fallback if pools are empty - still respect exclusion
    if not mid_nodes and not low_nodes:
        all_acc = accounts_df['account_id'].tolist()
        potential = [n for n in all_acc if n not in exclude_nodes]
        return np.random.choice(potential) if potential else np.random.choice(all_acc)
    
    if np.random.random() < 0.10 and mid_nodes:
        return np.random.choice(mid_nodes)
    else:
        return np.random.choice(low_nodes) if low_nodes else np.random.choice(mid_nodes)


def _apply_embedding(
    path: List[str],
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    exclude_nodes: Set[str]
) -> List[str]:
    """
    Inject legitimate intermediaries into a laundering path.
    - Ratio: config.EMBEDDED_RATIO
    - Constraint: At most 1 intermediary every 4 hops.
    - Constraint: No consecutive intermediaries.
    """
    if len(path) < 2:
        return path
        
    new_path = [path[0]]
    hops_since_last_inter = 100 # Infinity start
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Decide whether to inject between u and v
        # Ratio 0.23 means roughly every 4-5 edges
        # We increase the spacing threshold to 10 to restore cohesion
        can_inject = hops_since_last_inter >= 10
        
        if can_inject and np.random.random() < config.EMBEDDED_RATIO:
            inter = _get_legit_intermediary(accounts_df, graph, exclude_nodes)
            new_path.append(inter)
            hops_since_last_inter = 1 # We just added an edge to 'inter'
            new_path.append(v)
            hops_since_last_inter += 1 # Next edge is inter -> v
        else:
            new_path.append(v)
            hops_since_last_inter += 1
            
    return new_path


def inject_legitimate_structural_noise(
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    seed: int = None
) -> List[Dict]:
    """
    Inject 'Sham' motifs (chains, cycles, bursts) for legitimate accounts.
    Target: ~5% of nodes involved in motifs.
    Constraints: Lower density (1-2 txs/edge), wider windows (12-24h).
    """
    if seed is not None:
        np.random.seed(seed)
        
    noise_txs = []
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    all_accounts = list(accounts_df['account_id'])
    
    # Target 5% of nodes
    num_noise_nodes = int(len(all_accounts) * 0.05)
    noise_participants = set(np.random.choice(all_accounts, size=num_noise_nodes, replace=False))
    available = list(noise_participants)
    
    start_sim = datetime(2024, 1, 1)
    
    # 1. Sham Chains (Larger: 10-15 nodes)
    num_chains = num_noise_nodes // 40
    for _ in range(num_chains):
        if len(available) < 15: break
        c_len = np.random.randint(10, 16)
        chain = [available.pop() for _ in range(c_len)]
        start_date = start_sim + timedelta(days=float(np.random.randint(0, 80)))
        
        for i in range(len(chain)-1):
            for _ in range(np.random.randint(1, 4)):
                noise_txs.append({
                    'sender': chain[i], 'receiver': chain[i+1],
                    'amount': round(np.random.lognormal(8.0, 1.5), 2),
                    'timestamp': start_date + timedelta(hours=np.random.uniform(0, 12)),
                    'sender_country': account_dict[chain[i]]['country_code'],
                    'receiver_country': account_dict[chain[i+1]]['country_code'],
                    'laundering_flag': 0, 'campaign_id': None
                })

    # 2. Sham Cycles (Larger: 8-12 nodes)
    num_cycles = num_noise_nodes // 50
    for _ in range(num_cycles):
        if len(available) < 12: break
        c_len = np.random.randint(8, 13)
        cycle = [available.pop() for _ in range(c_len)]
        start_date = start_sim + timedelta(days=float(np.random.randint(0, 80)))
        for i in range(len(cycle)):
            for _ in range(np.random.randint(1, 4)):
                noise_txs.append({
                    'sender': cycle[i], 'receiver': cycle[(i+1)%len(cycle)],
                    'amount': round(np.random.lognormal(8.0, 1.5), 2),
                    'timestamp': start_date + timedelta(hours=np.random.uniform(0, 12)),
                    'sender_country': account_dict[cycle[i]]['country_code'],
                    'receiver_country': account_dict[cycle[(i+1)%len(cycle)]]['country_code'],
                    'laundering_flag': 0, 'campaign_id': None
                })

    # 3. Sham Clusters (Large Star/Smurf-mimic: 15-30 nodes)
    num_clusters = num_noise_nodes // 30
    for _ in range(num_clusters):
        if len(available) < 30: break
        center = available.pop()
        m_count = np.random.randint(15, 31)
        mules = [available.pop() for _ in range(m_count)]
        start_date = start_sim + timedelta(days=float(np.random.randint(0, 80)))
        # MIMIC Smurfing structure exactly but with legitimate flag
        for m in mules:
            for _ in range(np.random.randint(2, 5)):
                noise_txs.append({
                    'sender': center, 'receiver': m,
                    'amount': round(np.random.lognormal(config.AMOUNT_MEAN, config.AMOUNT_STD), 2),
                    'timestamp': start_date + timedelta(hours=np.random.uniform(0, 24)),
                    'sender_country': account_dict[center]['country_code'],
                    'receiver_country': account_dict[m]['country_code'],
                    'laundering_flag': 0, 'campaign_id': None
                })
            
    # 4. Dense Clusters (Mimicry)
    # Target: ~2.0% of population nodes. (Reduced noise to help F1)
    num_dense_nodes = int(len(all_accounts) * 0.02)
    dense_participants = set(np.random.choice(all_accounts, size=num_dense_nodes, replace=False))
    available_dense = list(dense_participants)
    
    num_clusters = num_dense_nodes // 8
    for _ in range(num_clusters):
        if len(available_dense) < 6: break
        c_size = np.random.randint(5, 11)
        cluster = [available_dense.pop() for _ in range(min(c_size, len(available_dense)))]
        
        # Match laundering edge density (e.g., each node connected to ~40-60% of others)
        num_internal_links = int(len(cluster) * (len(cluster)-1) * 0.5)
        start_date = start_sim + timedelta(days=float(np.random.randint(0, 80)))
        
        for _ in range(num_internal_links):
            u, v = np.random.choice(cluster, size=2, replace=False)
            # DILATED Timing: 1-5 days to distinguish from compressed laundering
            dilation_days = np.random.randint(1, 6)
            for _ in range(np.random.randint(1, 4)):
                noise_txs.append({
                    'sender': u, 'receiver': v,
                    'amount': round(np.random.lognormal(8.0, 1.5), 2),
                    'timestamp': start_date + timedelta(days=float(np.random.uniform(0, dilation_days))),
                    'sender_country': account_dict[u]['country_code'],
                    'receiver_country': account_dict[v]['country_code'],
                    'laundering_flag': 0, 'campaign_id': None
                })
            
    return noise_txs


def inject_cross_contamination(
    laundering_txs: List[Dict],
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    seed: int = None
) -> List[Dict]:
    """
    Inject bidirectional edges between laundering nodes and legitimate accounts.
    - 20-30% of laundering nodes interact with legitimate nodes.
    - 30-40% of these edges are legitimate -> laundering.
    """
    if seed is not None:
        np.random.seed(seed + 100) # Offset seed
        
    laundering_nodes = set([tx['sender'] for tx in laundering_txs] + [tx['receiver'] for tx in laundering_txs])
    legit_nodes = list(set(accounts_df['account_id']) - laundering_nodes)
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    
    contamination_txs = []
    
    # Track nodes by campaign
    nodes_by_campaign = {}
    for tx in laundering_txs:
        cid = tx['campaign_id']
        if cid not in nodes_by_campaign: nodes_by_campaign[cid] = set()
        nodes_by_campaign[cid].add(tx['sender'])
        nodes_by_campaign[cid].add(tx['receiver'])
        
    for cid, campaign_nodes in nodes_by_campaign.items():
        # REDUCED: Select 10-15% of nodes in this campaign for contamination
        target_nodes = list(np.random.choice(list(campaign_nodes), size=max(1, int(len(campaign_nodes) * np.random.uniform(0.10, 0.15))), replace=False))
        
        for laund_node in target_nodes:
            # 2-4 transactions for stronger drop in internal ratio
            for _ in range(np.random.randint(2, 5)):
                legit_node = np.random.choice(legit_nodes)
                
                # Bidirectional: 35% legit -> laund
                if np.random.random() < 0.35:
                    src, dst = legit_node, laund_node
                else:
                    src, dst = laund_node, legit_node
                    
                # Wider window (12-48h), regular amounts
                ts = datetime(2024, 1, 1) + timedelta(days=float(np.random.randint(0, 85)), hours=float(np.random.randint(0, 24)))
                
                contamination_txs.append({
                    'sender': src, 'receiver': dst,
                    'amount': round(np.random.lognormal(8.0, 1.5), 2),
                    'timestamp': ts + timedelta(hours=np.random.uniform(0, 48)),
                    'sender_country': account_dict[src]['country_code'],
                    'receiver_country': account_dict[dst]['country_code'],
                    'laundering_flag': 0, 'campaign_id': None # Legitimate link
                })
                
    return contamination_txs


def _generate_smurfing_campaign(
    campaign_id: int,
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    laundering_nodes: Set[str],
    start_date: datetime
) -> Tuple[List[Dict], Dict]:
    """
    Generate a Smurfing campaign with heavier transaction density.
    Target: 150-300 transactions per campaign
    Pattern: 2-3 origins -> 15-25 mules -> 2-3 sinks
    
    Args:
        campaign_id: Campaign identifier
        accounts_df: DataFrame with accounts
        graph: Structural graph
        laundering_nodes: Set of accounts involved in laundering
        start_date: Simulation start date
        
    Returns:
        Tuple of (transactions, metadata)
    """
    transactions = []
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    
    # Select participants from laundering nodes
    available = list(laundering_nodes)
    if len(available) < 25:
        # Fallback: use any accounts
        available = list(accounts_df['account_id'])
    
    num_origins = np.random.randint(*config.SMURFING_ORIGIN_COUNT_RANGE)
    num_mules = np.random.randint(*config.SMURFING_MULE_COUNT_RANGE)
    num_sinks = np.random.randint(*config.SMURFING_SINK_COUNT_RANGE)
    
    total_needed = num_origins + num_mules + num_sinks
    if len(available) < total_needed:
        num_mules = max(10, len(available) - num_origins - num_sinks)
    
    origins = np.random.choice(available, size=num_origins, replace=False).tolist()
    remaining = [a for a in available if a not in origins]
    mules = np.random.choice(remaining, size=min(num_mules, len(remaining)), replace=False).tolist()
    remaining = [a for a in remaining if a not in mules]
    sinks = np.random.choice(remaining, size=min(num_sinks, len(remaining)), replace=False).tolist()
    
    # Campaign timing with tight temporal compression
    campaign_start = start_date
    time_window_hours = np.random.uniform(*config.SMURFING_TIME_WINDOW_HOURS)
    
    participants = set(origins + mules + sinks)
    
    # Phase 1: Origins -> Mules (HEAVY density)
    for origin in origins:
        # Each origin sends to MOST mules (not just a few)
        num_mule_targets = max(8, int(len(mules) * np.random.uniform(0.5, 0.8)))
        mule_targets = np.random.choice(mules, size=min(num_mule_targets, len(mules)), replace=False)
        
        for mule in mule_targets:
            # NEW: Embedded Path
            route = _apply_embedding([origin, mule], accounts_df, graph, participants)
            # Add any newly discovered legit nodes to participants subset
            participants.update(route)
            
            # Each origin-mule pair generates MULTIPLE transactions
            num_txs = int(np.random.randint(2, 5) * config.SMURFING_TX_MULTIPLIER)
            
            for _ in range(num_txs):
                for step in range(len(route)-1):
                    src, dst = route[step], route[step+1]
                    amount = np.random.lognormal(mean=config.AMOUNT_MEAN, sigma=config.AMOUNT_STD)
                    time_offset = timedelta(hours=np.random.uniform(0, time_window_hours * 0.45))
                    timestamp = campaign_start + time_offset
                    
                    transactions.append({
                        'sender': src,
                        'receiver': dst,
                        'amount': round(amount, 2),
                        'timestamp': timestamp,
                        'sender_country': account_dict[src]['country_code'],
                        'receiver_country': account_dict[dst]['country_code'],
                        'laundering_flag': 1,
                        'campaign_id': campaign_id
                    })
    
    # Phase 2: Mules -> Sinks (HEAVY density)
    for mule in mules:
        # Each mule sends to 2-3 sinks
        if len(sinks) == 0:
            continue
        
        max_sinks = min(4, len(sinks) + 1)
        num_sink_targets = 2 if max_sinks <= 2 else np.random.randint(2, max_sinks)
        sink_targets = np.random.choice(sinks, size=min(num_sink_targets, len(sinks)), replace=True)
        
        for sink in sink_targets:
            # Multiple transactions per mule-sink pair
            num_txs_per_sink = int(np.ceil(config.SMURFING_TX_MULTIPLIER))  # Adjusted for density
            for _ in range(num_txs_per_sink):
                amount = np.random.lognormal(mean=config.AMOUNT_MEAN, sigma=config.AMOUNT_STD)
            # Second half of time window
            time_offset = timedelta(hours=np.random.uniform(time_window_hours * 0.5, time_window_hours))
            timestamp = campaign_start + time_offset
            
            transactions.append({
                'sender': mule,
                'receiver': sink,
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'sender_country': account_dict[mule]['country_code'],
                'receiver_country': account_dict[sink]['country_code'],
                'laundering_flag': 1,
                'campaign_id': campaign_id
            })
    
    # Phase 3: Cross-mule transactions (increased density)
    num_cross_mule = int(len(mules) * config.SMURFING_CROSS_MULE_RATIO * config.SMURFING_TX_MULTIPLIER * 2)
    for _ in range(num_cross_mule):
        if len(mules) < 2:
            break
        sender = np.random.choice(mules)
        receiver = np.random.choice([m for m in mules if m != sender])
        
        amount = np.random.lognormal(mean=8.0, sigma=1.1)
        time_offset = timedelta(hours=np.random.uniform(0, time_window_hours))
        timestamp = campaign_start + time_offset
        
        transactions.append({
            'sender': sender,
            'receiver': receiver,
            'amount': round(amount, 2),
            'timestamp': timestamp,
            'sender_country': account_dict[sender]['country_code'],
            'receiver_country': account_dict[receiver]['country_code'],
            'laundering_flag': 1,
            'campaign_id': campaign_id
        })
    
    # Phase 4: Noise transactions (20-30% of current count)
    num_noise = int(len(transactions) * config.SMURFING_NOISE_RATIO)
    for _ in range(num_noise):
        sender = np.random.choice(mules + origins)
        # Random receiver (not in campaign)
        all_accounts = list(accounts_df['account_id'])
        receiver = np.random.choice(all_accounts)
        
        if receiver in participants:
            continue
        
        amount = np.random.lognormal(mean=7.5, sigma=1.5)
        time_offset = timedelta(hours=np.random.uniform(0, time_window_hours))
        timestamp = campaign_start + time_offset
        
        transactions.append({
            'sender': sender,
            'receiver': receiver,
            'amount': round(amount, 2),
            'timestamp': timestamp,
            'sender_country': account_dict[sender]['country_code'],
            'receiver_country': account_dict[receiver]['country_code'],
            'laundering_flag': 1,
            'campaign_id': campaign_id
        })
    
    # Compute campaign metadata
    timestamps = [tx['timestamp'] for tx in transactions]
    time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60 if timestamps else 0
    
    degrees = dict(graph.degree())
    avg_degree = np.mean([degrees.get(acc, 0) for acc in participants])
    
    metadata = {
        'campaign_id': campaign_id,
        'typology': 'smurfing',
        'num_nodes': len(participants),
        'num_transactions': len(transactions),
        'time_span_minutes': time_span_minutes,
        'avg_degree': avg_degree,
        'participants': list(participants)
    }
    
    return transactions, metadata


def _generate_layering_campaign(
    campaign_id: int,
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    laundering_nodes: Set[str],
    start_date: datetime
) -> Tuple[List[Dict], Dict]:
    """
    Generate a Layering campaign with heavier transaction density.
    Target: 120-250 transactions per campaign
    Pattern: Sequential chain of 4-7 nodes with multiple transfers per hop
    
    Args:
        campaign_id: Campaign identifier
        accounts_df: DataFrame with accounts
        graph: Structural graph
        laundering_nodes: Set of laundering accounts
        start_date: Simulation start date
        
    Returns:
        Tuple of (transactions, metadata)
    """
    transactions = []
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    
    # Select chain participants
    available = list(laundering_nodes)
    if len(available) < 15:
        available = list(accounts_df['account_id'])
    
    chain_length = np.random.randint(*config.LAYERING_CHAIN_LENGTH_RANGE)
    chain = np.random.choice(available, size=min(chain_length, len(available)), replace=False).tolist()
    
    # Campaign timing with TIGHT temporal compression (1-3 hours total)
    campaign_start = start_date
    time_window_hours = np.random.uniform(*config.LAYERING_TIME_WINDOW_HOURS)
    time_per_hop = time_window_hours / len(chain) * 60  # minutes per hop
    
    # Initial amount
    amount = np.random.lognormal(mean=config.AMOUNT_MEAN, sigma=config.AMOUNT_STD)
    
    participants = set(chain)
    current_time = campaign_start
    
    # Generate chain transactions with MULTIPLE transfers per hop
    for i in range(len(chain) - 1):
        sender = chain[i]
        receiver = chain[i + 1]
        
        # NEW: Embedded Path
        route = _apply_embedding([sender, receiver], accounts_df, graph, participants)
        participants.update(route)
            
        # Primary transfers (MULTIPLE per hop for density)
        num_primary = int(np.ceil(config.LAYERING_TX_MULTIPLIER))
        for j in range(num_primary):
            for step in range(len(route)-1):
                s, r = route[step], route[step+1]
                time_offset = timedelta(minutes=np.random.uniform(0, time_per_hop * 0.8))
                tx_time = current_time + time_offset
                tx_amount = amount * np.random.uniform(0.9, 1.1)
                
                transactions.append({
                    'sender': s,
                    'receiver': r,
                    'amount': round(tx_amount, 2),
                    'timestamp': tx_time,
                    'sender_country': account_dict[s]['country_code'],
                    'receiver_country': account_dict[r]['country_code'],
                    'laundering_flag': 1,
                    'campaign_id': campaign_id
                })
        
        # Amount shrinkage for next hop
        amount *= np.random.uniform(0.88, 0.95)
        
        # Move to next hop time window
        current_time += timedelta(minutes=time_per_hop)
    
    # Add side transfers from middle nodes
    if len(chain) >= 3:
        min_side = config.LAYERING_SIDE_TRANSFERS_RANGE[0]
        max_side = config.LAYERING_SIDE_TRANSFERS_RANGE[1]
        num_side_transfers = min_side if max_side <= min_side else np.random.randint(min_side, max_side + 1)
        
        for _ in range(num_side_transfers):
            # Select middle node (only if there are middle nodes)
            if len(chain) < 3:
                break
            middle_idx = 1 if len(chain) == 3 else np.random.randint(1, len(chain) - 1)
            sender = chain[middle_idx]
            
            # Target: another node in chain or external
            if np.random.random() < 0.5 and len(chain) > 3:
                # Internal chain node
                receiver = np.random.choice([n for n in chain if n != sender])
            else:
                # External node
                other_accounts = [a for a in available if a not in chain]
                if other_accounts:
                    receiver = np.random.choice(other_accounts)
                    participants.add(receiver)
                else:
                    continue
            
            # Multiple side transfers per pair (increased)
            num_txs = int(np.ceil(config.LAYERING_TX_MULTIPLIER))  # Adjusted for density
            for _ in range(num_txs):
                side_amount = amount * np.random.uniform(0.2, 0.4)
                side_time = campaign_start + timedelta(hours=np.random.uniform(0, time_window_hours))
                
                transactions.append({
                    'sender': sender,
                    'receiver': receiver,
                    'amount': round(side_amount, 2),
                    'timestamp': side_time,
                    'sender_country': account_dict[sender]['country_code'],
                    'receiver_country': account_dict[receiver]['country_code'],
                    'laundering_flag': 1,
                    'campaign_id': campaign_id
                })
    
    # Add diversion edges (branching complexity)
    num_diversions = max(2, len(chain) // 3)
    for _ in range(num_diversions):
        diversion_node = np.random.choice(chain)
        other_accounts = [a for a in available if a not in participants]
        if other_accounts:
            diversion_target = np.random.choice(other_accounts)
            participants.add(diversion_target)
            
            # Multiple diversion transactions (increased)
            num_div_txs = int(np.ceil(config.LAYERING_TX_MULTIPLIER))  # Adjusted for density
            for _ in range(num_div_txs):
                div_amount = amount * np.random.uniform(0.15, 0.35)
                div_time = campaign_start + timedelta(hours=np.random.uniform(0, time_window_hours))
                
                transactions.append({
                    'sender': diversion_node,
                    'receiver': diversion_target,
                    'amount': round(div_amount, 2),
                    'timestamp': div_time,
                    'sender_country': account_dict[diversion_node]['country_code'],
                    'receiver_country': account_dict[diversion_target]['country_code'],
                    'laundering_flag': 1,
                    'campaign_id': campaign_id
                })
    
    # Compute campaign metadata
    timestamps = [tx['timestamp'] for tx in transactions]
    time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60 if timestamps else 0
    
    degrees = dict(graph.degree())
    avg_degree = np.mean([degrees.get(acc, 0) for acc in participants])
    
    metadata = {
        'campaign_id': campaign_id,
        'typology': 'layering',
        'num_nodes': len(participants),
        'num_transactions': len(transactions),
        'time_span_minutes': time_span_minutes,
        'avg_degree': avg_degree,
        'participants': list(participants)
    }
    
    return transactions, metadata


def _generate_circular_campaign(
    campaign_id: int,
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    laundering_nodes: Set[str],
    start_date: datetime
) -> Tuple[List[Dict], Dict]:
    """
    Generate a Circular campaign with heavier transaction density.
    Target: 100-200 transactions per campaign
    Pattern: 5-7 node cycle with multiple transfers and branches
    
    Args:
        campaign_id: Campaign identifier
        accounts_df: DataFrame with accounts
        graph: Structural graph
        laundering_nodes: Set of laundering accounts
        start_date: Simulation start date
        
    Returns:
        Tuple of (transactions, metadata)
    """
    transactions = []
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    
    # Select cycle participants
    available = list(laundering_nodes)
    if len(available) < 12:
        available = list(accounts_df['account_id'])
    
    cycle_length = np.random.randint(*config.CIRCULAR_CYCLE_LENGTH_RANGE)
    cycle = np.random.choice(available, size=min(cycle_length, len(available)), replace=False).tolist()
    
    # Campaign timing with moderate temporal compression
    campaign_start = start_date
    time_window_hours = np.random.uniform(*config.CIRCULAR_TIME_WINDOW_HOURS)
    
    # Initial amount
    amount = np.random.lognormal(mean=config.AMOUNT_MEAN, sigma=config.AMOUNT_STD)
    
    participants = set(cycle)
    
    # Generate cycle transactions (MULTIPLE rounds)
    num_rounds = np.random.randint(3, 6)  # Cycle multiple times
    for round_num in range(num_rounds):
        for i in range(len(cycle)):
            sender = cycle[i]
            receiver = cycle[(i + 1) % len(cycle)]
            
            # NEW: Embedded Path
            route = _apply_embedding([sender, receiver], accounts_df, graph, participants)
            participants.update(route)
                
            # Multiple transactions per edge per round
            num_txs = int(np.ceil(config.CIRCULAR_TX_MULTIPLIER))
            for _ in range(num_txs):
                for step in range(len(route)-1):
                    s, r = route[step], route[step+1]
                    tx_amount = amount * np.random.uniform(0.85, 1.15)
                    time_offset = timedelta(hours=np.random.uniform(
                        round_num * (time_window_hours / num_rounds),
                        (round_num + 1) * (time_window_hours / num_rounds)
                    ))
                    tx_time = campaign_start + time_offset
                    
                    transactions.append({
                        'sender': s,
                        'receiver': r,
                        'amount': round(tx_amount, 2),
                        'timestamp': tx_time,
                        'sender_country': account_dict[s]['country_code'],
                        'receiver_country': account_dict[r]['country_code'],
                        'laundering_flag': 1,
                        'campaign_id': campaign_id
                    })
    
    # Add branch transfers (each node sends to external targets)
    for node in cycle:
        num_branches = config.CIRCULAR_BRANCH_COUNT
        
        for _ in range(num_branches):
            # External target
            other_accounts = [a for a in available if a not in participants]
            if not other_accounts:
                break
            
            branch_target = np.random.choice(other_accounts)
            participants.add(branch_target)
            
            # Multiple transactions per branch (increased)
            num_branch_txs = int(np.ceil(config.CIRCULAR_TX_MULTIPLIER))  # Adjusted for density
            for _ in range(num_branch_txs):
                branch_amount = amount * np.random.uniform(0.2, 0.4)
                branch_time = campaign_start + timedelta(hours=np.random.uniform(0, time_window_hours))
                
                transactions.append({
                    'sender': node,
                    'receiver': branch_target,
                    'amount': round(branch_amount, 2),
                    'timestamp': branch_time,
                    'sender_country': account_dict[node]['country_code'],
                    'receiver_country': account_dict[branch_target]['country_code'],
                    'laundering_flag': 1,
                    'campaign_id': campaign_id
                })
    
    # Add cross-cycle noise edges
    num_cross_cycle = max(2, int(len(cycle) * 0.4))
    for _ in range(num_cross_cycle):
        if len(cycle) < 2:
            break
        
        sender = np.random.choice(cycle)
        receiver = np.random.choice([n for n in cycle if n != sender])
        
        # Multiple transactions per cross edge (increased)
        num_txs = int(np.ceil(config.CIRCULAR_TX_MULTIPLIER))  # Adjusted for density
        for _ in range(num_txs):
            cross_amount = amount * np.random.uniform(0.3, 0.7)
            cross_time = campaign_start + timedelta(hours=np.random.uniform(0, time_window_hours))
            
            transactions.append({
                'sender': sender,
                'receiver': receiver,
                'amount': round(cross_amount, 2),
                'timestamp': cross_time,
                'sender_country': account_dict[sender]['country_code'],
                'receiver_country': account_dict[receiver]['country_code'],
                'laundering_flag': 1,
                'campaign_id': campaign_id
            })
    
    # Compute campaign metadata
    timestamps = [tx['timestamp'] for tx in transactions]
    time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60 if timestamps else 0
    
    degrees = dict(graph.degree())
    avg_degree = np.mean([degrees.get(acc, 0) for acc in participants])
    
    metadata = {
        'campaign_id': campaign_id,
        'typology': 'circular',
        'num_nodes': len(participants),
        'num_transactions': len(transactions),
        'time_span_minutes': time_span_minutes,
        'avg_degree': avg_degree,
        'participants': list(participants)
    }
    
    return transactions, metadata
