"""
Temporal transaction engine.
Simulates realistic transaction patterns over time.
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from . import config


def generate_baseline_transactions(
    accounts_df: pd.DataFrame,
    graph: nx.DiGraph,
    seed: int = None
) -> List[Dict]:
    """
    Generate baseline (legitimate) transactions over simulation period.
    
    Args:
        accounts_df: DataFrame with account information
        graph: NetworkX DiGraph representing structural affinity
        seed: Random seed for reproducibility
        
    Returns:
        List of transaction dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    transactions = []
    account_dict = accounts_df.set_index('account_id').to_dict('index')
    
    # Simulation start date
    start_date = datetime(2024, 1, 1)
    
    for day in range(config.SIMULATION_DAYS):
        current_date = start_date + timedelta(days=day)
        
        # Determine day type and volume multiplier
        multiplier = _get_volume_multiplier(day, current_date)
        
        # Expected transactions for this day
        expected_tx = int(config.BASE_TX_PER_DAY * multiplier)
        
        # Generate transactions
        day_transactions = _generate_day_transactions(
            day=day,
            current_date=current_date,
            expected_tx=expected_tx,
            accounts_df=accounts_df,
            account_dict=account_dict,
            graph=graph
        )
        
        transactions.extend(day_transactions)
    
    return transactions


def _get_volume_multiplier(day: int, date: datetime) -> float:
    """
    Determine volume multiplier based on day type.
    
    Args:
        day: Day number (0-indexed)
        date: Current date
        
    Returns:
        Volume multiplier
    """
    weekday = date.weekday()
    
    # Check if salary day (last working day of month)
    is_salary_day = _is_salary_day(date)
    
    if is_salary_day:
        return config.SALARY_DAY_MULTIPLIER
    elif weekday >= 5:  # Weekend (Saturday=5, Sunday=6)
        return config.WEEKEND_MULTIPLIER
    else:
        return config.WEEKDAY_MULTIPLIER


def _is_salary_day(date: datetime) -> bool:
    """
    Check if date is a salary day (last working day of month).
    
    Args:
        date: Date to check
        
    Returns:
        True if salary day
    """
    # Simple heuristic: 28th-31st of month with some probability
    if date.day >= 28:
        return np.random.random() < 0.3
    return False


def _generate_day_transactions(
    day: int,
    current_date: datetime,
    expected_tx: int,
    accounts_df: pd.DataFrame,
    account_dict: Dict,
    graph: nx.DiGraph
) -> List[Dict]:
    """
    Generate transactions for a single day.
    
    Args:
        day: Day number
        current_date: Current date
        expected_tx: Expected number of transactions
        accounts_df: DataFrame with accounts
        account_dict: Dictionary mapping account_id to account info
        graph: Structural affinity graph
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    # Get active accounts (created on or before this day)
    active_accounts = accounts_df[accounts_df['creation_day'] <= day].copy()
    
    if len(active_accounts) == 0:
        return transactions
    
    # Activity probabilities proportional to activity_score
    activity_probs = active_accounts['activity_score'].values
    activity_probs = activity_probs / activity_probs.sum()
    
    # Sample sending accounts
    num_senders = min(expected_tx, len(active_accounts))
    senders = np.random.choice(
        active_accounts['account_id'].values,
        size=num_senders,
        replace=True,
        p=activity_probs
    )
    
    # Generate transactions
    for sender_id in senders:
        # Get potential receivers (structural neighbors)
        neighbors = list(graph.successors(sender_id))
        
        if len(neighbors) == 0:
            # Fallback: random receiver from active accounts
            receiver_id = np.random.choice(active_accounts['account_id'].values)
        else:
            receiver_id = np.random.choice(neighbors)
        
        # Skip self-transactions
        if sender_id == receiver_id:
            continue
        
        # Generate amount (lognormal distribution)
        amount = np.random.lognormal(
            mean=config.AMOUNT_MEAN,
            sigma=config.AMOUNT_STD
        )
        amount = round(amount, 2)
        
        # Generate timestamp (biased toward peak hours)
        timestamp = _generate_timestamp(current_date)
        
        # Get sender and receiver countries
        sender_country = account_dict[sender_id]['country_code']
        receiver_country = account_dict[receiver_id]['country_code']
        
        transaction = {
            'sender': sender_id,
            'receiver': receiver_id,
            'amount': amount,
            'timestamp': timestamp,
            'sender_country': sender_country,
            'receiver_country': receiver_country,
            'laundering_flag': 0,
            'campaign_id': None
        }
        
        transactions.append(transaction)
    
    return transactions


def _generate_timestamp(date: datetime) -> datetime:
    """
    Generate timestamp for a transaction, biased toward peak hours.
    
    Args:
        date: Date for the transaction
        
    Returns:
        Timestamp with time of day
    """
    # Bias toward peak hours (8am - 10pm)
    if np.random.random() < (1 - config.NIGHT_TRAFFIC_RATIO):
        # Peak hours
        hour = np.random.randint(config.PEAK_HOURS_START, config.PEAK_HOURS_END)
    else:
        # Night hours
        night_hours = list(range(0, config.PEAK_HOURS_START)) + \
                     list(range(config.PEAK_HOURS_END, 24))
        hour = np.random.choice(night_hours)
    
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    
    return datetime(date.year, date.month, date.day, hour, minute, second)
