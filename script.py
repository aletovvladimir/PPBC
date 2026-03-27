import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters
FEDERATED_METHODS = [
    "ppbc"
]
BASE_PARAMS = ['federated_method=ppbc',
    'federated_method.iterations=5',
    'federated_params.amount_of_clients=200',
    'federated_params.communication_rounds=100',
    'federated_params.round_epochs=25',
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=True",
]

cmd = ["nohup", "python", "train.py"] + BASE_PARAMS
output_path = f'ppbc_rare_25le_5iters.txt'
cmd_str = " ".join(cmd) + f" > {output_path} &"
# Print and execute
subprocess.run(cmd_str, shell=True, check=True)
