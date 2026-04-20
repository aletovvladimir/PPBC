import subprocess
import os
import argparse
from utils.dirichlet import DirichletDistribution

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters
FEDERATED_METHODS = [
    "ppbc",
    "ferdavg"
]

alphas = [5]

BASE_PARAMS = ['federated_method=ppbc',
    'federated_params.communication_rounds=30',
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=True",
]

EPOCH_METHODS = ['random']
ITER_METHODS = ['random']
# for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):

#     for alpha in alphas:

#         params = [ f"dirichlet_alpha={alpha}", 
#         f'federated_method.epoch_method={e_m}',
#         f'federated_method.iter_method={i_m}',
#         'federated_method.iterations=3']

#         cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
#         output_path = f'ppbc_{e_m}_{i_m}_{alpha}.txt'
#         cmd_str = " ".join(cmd) + f" > {output_path}"
#         # Print and execute
#         print(f"Running setup: {e_m}+{i_m}", flush=True)
#         subprocess.run(cmd_str, shell=True, check=True)

BASE_PARAMS = ['federated_method=fedavg',
    'federated_params.communication_rounds=30',
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=True",]

for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):

    for alpha in alphas:

        params = [ f"dirichlet_alpha={alpha}",] 
        # f'federated_method.epoch_method={e_m}',
        # f'federated_method.iter_method={i_m}',
        # 'federated_method.iterations=3']

        cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
        output_path = f'fedavg_{e_m}_{i_m}_{alpha}.txt'
        cmd_str = " ".join(cmd) + f" > {output_path}"
        # Print and execute
        print(f"Running setup: {e_m}+{i_m}", flush=True)
        subprocess.run(cmd_str, shell=True, check=True)
