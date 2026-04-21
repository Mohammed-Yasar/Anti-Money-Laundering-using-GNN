"""
Account generation module.
Creates synthetic financial accounts with realistic activity profiles.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from . import config


def generate_accounts(seed: int = None) -> pd.DataFrame:
    """
    Generate synthetic financial accounts with activity scores.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: account_id, country_code, creation_day, activity_score
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_accounts = config.NUM_ACCOUNTS
    
    # Generate account IDs
    account_ids = [f"ACC{str(i).zfill(6)}" for i in range(num_accounts)]
    
    # Assign country codes
    country_codes = np.random.choice(
        config.COUNTRIES, 
        size=num_accounts, 
        p=_get_country_probabilities()
    )
    
    # Assign creation days (within first 30 days of simulation)
    creation_days = np.random.randint(0, min(30, config.SIMULATION_DAYS), size=num_accounts)
    
    # Sample activity scores from lognormal distribution
    activity_scores = np.random.lognormal(
        mean=config.ACTIVITY_MEAN,
        sigma=config.ACTIVITY_STD,
        size=num_accounts
    )
    
    # Create DataFrame
    accounts_df = pd.DataFrame({
        'account_id': account_ids,
        'country_code': country_codes,
        'creation_day': creation_days,
        'activity_score': activity_scores
    })
    
    return accounts_df


def _get_country_probabilities() -> list:
    """
    Get probability distribution for country assignment.
    
    Returns:
        List of probabilities corresponding to COUNTRIES
    """
    # Realistic distribution: US dominant, then IN, UK, SG, AE
    probs = [0.40, 0.25, 0.20, 0.10, 0.05]
    return probs


def save_accounts(accounts_df: pd.DataFrame, output_path: str) -> None:
    """
    Save accounts DataFrame to CSV.
    
    Args:
        accounts_df: DataFrame containing account information
        output_path: Path to save CSV file
    """
    accounts_df.to_csv(output_path, index=False)
    print(f"Saved {len(accounts_df)} accounts to {output_path}")


def load_accounts(input_path: str) -> pd.DataFrame:
    """
    Load accounts DataFrame from CSV.
    
    Args:
        input_path: Path to CSV file
        
    Returns:
        DataFrame with account information
    """
    return pd.read_csv(input_path)
