"""
Configuration file for AML transaction generator.
All parameters are defined here - no hard-coded values elsewhere.
"""

# Random seed for reproducibility
SEED = 42

# Account generation parameters
NUM_ACCOUNTS = 40000
HUB_RATIO = 0.08  # Proportion of high-activity hub accounts
DORMANT_RATIO = 0.10  # Proportion of low-activity dormant accounts

# Temporal simulation parameters
SIMULATION_DAYS = 90
BASE_TX_PER_DAY = 4500

# Laundering parameters
LAUNDERING_CAMPAIGNS = 350  # Increased further for 1% volume with 0.05 intensity
LAUNDERING_NODE_RATIO = 0.12  # More nodes to absorb reuse

# Laundering campaign density parameters
SMURFING_ORIGIN_COUNT_RANGE = (2, 3)
SMURFING_MULE_COUNT_RANGE = (15, 25)
SMURFING_SINK_COUNT_RANGE = (2, 3)
SMURFING_CROSS_MULE_RATIO = 0.25
SMURFING_NOISE_RATIO = 0.01  # Minimal noise for maximum cohesion

LAYERING_CHAIN_LENGTH_RANGE = (4, 7)  # Increased from 3-6
LAYERING_SIDE_TRANSFERS_RANGE = (1, 2)

CIRCULAR_CYCLE_LENGTH_RANGE = (5, 7)  # Increased from 4-6
CIRCULAR_BRANCH_COUNT = 2  # Branch transfers per node

# Temporal compression for laundering (in hours)
SMURFING_TIME_WINDOW_HOURS = (4, 12)
LAYERING_TIME_WINDOW_HOURS = (4, 12)
CIRCULAR_TIME_WINDOW_HOURS = (4, 12)

# Geographic distribution
COUNTRIES = ["US", "IN", "UK", "SG", "AE"]

# Network structure parameters
BARABASI_ALBERT_M = 3  # Number of edges to attach from new node in BA model

# Transaction amount parameters (lognormal distribution)
AMOUNT_MEAN = 8.0  # ln(amount) mean
AMOUNT_STD = 1.5   # ln(amount) standard deviation

# Activity score parameters (lognormal distribution)
ACTIVITY_MEAN = 0.0
ACTIVITY_STD = 1.2

# Volume multipliers by day type
WEEKDAY_MULTIPLIER = 1.0
WEEKEND_MULTIPLIER = 0.7
SALARY_DAY_MULTIPLIER = 1.8

# Time of day distribution (hour of day)
PEAK_HOURS_START = 8
PEAK_HOURS_END = 22
NIGHT_TRAFFIC_RATIO = 0.05

# Laundering typology transaction multipliers (Fine-tuned to hit 2.5 density)
SMURFING_TX_MULTIPLIER = 0.05
LAYERING_TX_MULTIPLIER = 0.05
CIRCULAR_TX_MULTIPLIER = 0.05

# Validation thresholds
MAX_LAUNDERING_RATIO = 0.035  # Warn if laundering exceeds 3.5%
MIN_LAUNDERING_RATIO = 0.02   # Minimum expected laundering ratio (2%)
DEGREE_THRESHOLD_TEST = 150   # Degree threshold for structural validation

# Calibration parameters for realistic laundering
EMBEDDED_RATIO = 0.15  # Slightly reduced as per Fix 2
MIN_INTERNAL_RATIO = 0.45
MAX_INTERNAL_RATIO = 0.65
MAX_DENSITY_RATIO = 2.5
REUSE_RATE_CAP = 0.20
CONTAMINATION_RATE = 0.05  # Reduced to restore cohesion
CLUSTER_RATE = 0.025
