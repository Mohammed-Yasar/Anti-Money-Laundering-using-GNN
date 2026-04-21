"""
Transaction builder module.
Merges baseline and laundering transactions into final dataset.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime


def build_final_dataset(
    baseline_txs: List[Dict],
    laundering_txs: List[Dict]
) -> pd.DataFrame:
    """
    Merge baseline and laundering transactions into final dataset.
    
    Args:
        baseline_txs: List of baseline transaction dictionaries
        laundering_txs: List of laundering transaction dictionaries
        
    Returns:
        DataFrame with all transactions
    """
    # Combine all transactions
    all_txs = baseline_txs + laundering_txs
    
    # Convert to DataFrame
    df = pd.DataFrame(all_txs)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add unique transaction IDs
    df['transaction_id'] = [f"TX{str(i).zfill(8)}" for i in range(len(df))]
    
    # Ensure no exact duplicate timestamps (add microseconds if needed)
    df = _resolve_duplicate_timestamps(df)
    
    # Reorder columns
    column_order = [
        'transaction_id',
        'sender',
        'receiver',
        'amount',
        'timestamp',
        'sender_country',
        'receiver_country',
        'laundering_flag',
        'campaign_id'
    ]
    
    df = df[column_order]
    
    return df


def _resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve duplicate timestamps by adding microseconds.
    
    Args:
        df: DataFrame with transactions
        
    Returns:
        DataFrame with unique timestamps
    """
    df = df.copy()
    
    # Group by timestamp and add microseconds to duplicates
    timestamp_counts = df.groupby('timestamp').size()
    duplicates = timestamp_counts[timestamp_counts > 1].index
    
    for ts in duplicates:
        mask = df['timestamp'] == ts
        duplicate_indices = df[mask].index
        
        # Add microseconds to each duplicate
        for i, idx in enumerate(duplicate_indices):
            if i > 0:
                df.at[idx, 'timestamp'] = df.at[idx, 'timestamp'] + pd.Timedelta(microseconds=i)
    
    return df


def save_transactions(df: pd.DataFrame, output_path: str) -> None:
    """
    Save transactions DataFrame to CSV.
    
    Args:
        df: DataFrame with transactions
        output_path: Path to save CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions to {output_path}")


def compute_dataset_statistics(df: pd.DataFrame, campaign_metadata: List[Dict] = None) -> Dict[str, any]:
    """
    Compute statistics about the generated dataset.
    
    Args:
        df: DataFrame with transactions
        campaign_metadata: List of campaign metadata dictionaries
        
    Returns:
        Dictionary with statistics
    """
    total_txs = len(df)
    laundering_txs = df['laundering_flag'].sum()
    laundering_ratio = laundering_txs / total_txs if total_txs > 0 else 0
    
    # Compute temporal deltas
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_delta'] = df_sorted['timestamp'].diff().dt.total_seconds() / 60  # minutes
    
    legit_deltas = df_sorted[df_sorted['laundering_flag'] == 0]['time_delta'].dropna()
    launder_deltas = df_sorted[df_sorted['laundering_flag'] == 1]['time_delta'].dropna()
    
    median_legit_delta = legit_deltas.median() if len(legit_deltas) > 0 else 0
    median_launder_delta = launder_deltas.median() if len(launder_deltas) > 0 else 0
    
    stats = {
        'total_transactions': total_txs,
        'baseline_transactions': total_txs - laundering_txs,
        'laundering_transactions': laundering_txs,
        'laundering_ratio': laundering_ratio,
        'unique_senders': df['sender'].nunique(),
        'unique_receivers': df['receiver'].nunique(),
        'total_volume': df['amount'].sum(),
        'avg_amount': df['amount'].mean(),
        'median_amount': df['amount'].median(),
        'num_campaigns': df['campaign_id'].nunique() - 1,  # Subtract 1 for None
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'median_legit_time_delta_min': median_legit_delta,
        'median_launder_time_delta_min': median_launder_delta
    }
    
    # Add campaign statistics if available
    if campaign_metadata:
        campaign_sizes = [c['num_transactions'] for c in campaign_metadata]
        stats['max_campaign_size'] = max(campaign_sizes) if campaign_sizes else 0
        stats['min_campaign_size'] = min(campaign_sizes) if campaign_sizes else 0
        stats['avg_campaign_size'] = np.mean(campaign_sizes) if campaign_sizes else 0
    
    return stats


def print_dataset_summary(stats: Dict[str, any]) -> None:
    """
    Print summary statistics of the dataset.
    
    Args:
        stats: Dictionary with statistics
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Transactions: {stats['total_transactions']:,}")
    print(f"  - Baseline (Legitimate): {stats['baseline_transactions']:,}")
    print(f"  - Laundering: {stats['laundering_transactions']:,}")
    print(f"Laundering Ratio: {stats['laundering_ratio']:.2%}")
    print(f"\nUnique Senders: {stats['unique_senders']:,}")
    print(f"Unique Receivers: {stats['unique_receivers']:,}")
    print(f"\nTotal Volume: ${stats['total_volume']:,.2f}")
    print(f"Average Amount: ${stats['avg_amount']:,.2f}")
    print(f"Median Amount: ${stats['median_amount']:,.2f}")
    print(f"\nNumber of Laundering Campaigns: {stats['num_campaigns']}")
    print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    
    # Temporal compression statistics
    print(f"\nTemporal Characteristics:")
    print(f"  - Median legitimate tx time delta: {stats['median_legit_time_delta_min']:.2f} min")
    print(f"  - Median laundering tx time delta: {stats['median_launder_time_delta_min']:.2f} min")
    if stats['median_legit_time_delta_min'] > 0:
        compression_ratio = stats['median_launder_time_delta_min'] / stats['median_legit_time_delta_min']
        print(f"  - Temporal compression ratio: {compression_ratio:.2f}x")
    
    # Campaign size statistics
    if 'max_campaign_size' in stats:
        print(f"\nCampaign Size Range:")
        print(f"  - Max: {stats['max_campaign_size']} transactions")
        print(f"  - Min: {stats['min_campaign_size']} transactions")
        print(f"  - Avg: {stats['avg_campaign_size']:.1f} transactions")
    
    print("=" * 60 + "\n")
