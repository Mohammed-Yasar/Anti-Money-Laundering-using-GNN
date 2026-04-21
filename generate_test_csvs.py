import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Create a directory to hold the test CSVs
output_dir = "test_csvs"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Invalid Schema CSV (Missing the 'amount' column)
# ---------------------------------------------------------
invalid_data = {
    "transaction_id": [f"TXN_INV_{i}" for i in range(10)],
    "sender": [f"ACC_S_{i}" for i in range(10)],
    "receiver": [f"ACC_R_{i}" for i in range(10)],
    # 'amount' is intentionally missing
    "timestamp": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(10)]
}
pd.DataFrame(invalid_data).to_csv(f"{output_dir}/1_invalid_schema.csv", index=False)


# ---------------------------------------------------------
# 2. Small Dataset CSV (Fails Data Quality Checks)
# ---------------------------------------------------------
# Needs: < 5000 transactions, < 1000 nodes, low avg and max degree
small_data = {
    "transaction_id": [f"TXN_SML_{i}" for i in range(500)],
    "sender": [f"ACC_{random.randint(1, 300)}" for _ in range(500)],
    "receiver": [f"ACC_{random.randint(1, 300)}" for _ in range(500)],
    "amount": [round(random.uniform(10, 1000), 2) for _ in range(500)],
    "timestamp": [(datetime.now() - timedelta(minutes=i)).isoformat() for i in range(500)]
}
pd.DataFrame(small_data).to_csv(f"{output_dir}/2_small_dataset.csv", index=False)


# ---------------------------------------------------------
# 3. Good Dataset CSV (Passes All Data Quality Checks)
# ---------------------------------------------------------
# Needs: >= 5000 transactions, >= 1000 nodes, max_degree >= 10, avg_degree >= 2
num_tx = 6000
senders = []
receivers = []

for _ in range(num_tx):
    # Introduce some "hub" accounts to bump the max degree > 10
    if random.random() < 0.1:
        senders.append(f"HUB_{random.randint(1, 5)}") 
    else:
        senders.append(f"ACC_{random.randint(1, 1500)}")
    
    if random.random() < 0.1:
        receivers.append(f"HUB_{random.randint(6, 10)}")
    else:
        receivers.append(f"ACC_{random.randint(1, 1500)}")

good_data = {
    "transaction_id": [f"TXN_GOOD_{i}" for i in range(num_tx)],
    "sender": senders,
    "receiver": receivers,
    "amount": [round(random.uniform(10, 5000), 2) for _ in range(num_tx)],
    "timestamp": [(datetime.now() - timedelta(minutes=i)).isoformat() for i in range(num_tx)]
}
pd.DataFrame(good_data).to_csv(f"{output_dir}/3_good_dataset.csv", index=False)

print(f"✅ Successfully generated 3 test CSV files in ./{output_dir}/")
