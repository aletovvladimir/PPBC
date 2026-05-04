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

alphas = [0.1, 0.5, 1]

# BASE_PARAMS = ['federated_method=ppbc',
#     'federated_params.communication_rounds=125',
#     "training_params.batch_size=32",
#     "federated_params.print_client_metrics=False",
# ]

# EPOCH_METHODS = ['gradient_norms', 'gradient_angles', 'loss']
# ITER_METHODS = ['loss', 'random', 'gradient_norms']
# for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):

#     for alpha in alphas:

#         params = [ f"dirichlet_alpha={alpha}", 
#         f'federated_method.epoch_method={e_m}',
#         f'federated_method.iter_method={i_m}',
#         'federated_method.iterations=5']

#         cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
#         output_path = f'ppbc_{e_m}_{i_m}_{alpha}_200clients.txt'
#         cmd_str = " ".join(cmd) + f" > {output_path}"
#         # Print and execute
#         print(f"Running setup: {e_m}+{i_m}", flush=True)
#         subprocess.run(cmd_str, shell=True, check=True)

# BASE_PARAMS = ['federated_method=fedavg',
#     'federated_params.communication_rounds=125',
#     "training_params.batch_size=32",
#     "federated_params.print_client_metrics=False",]

# # for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):

# for alpha in alphas:

#     params = [ f"dirichlet_alpha={alpha}",] 
#     # f'federated_method.epoch_method={e_m}',
#     # f'federated_method.iter_method={i_m}',
#     # 'federated_method.iterations=3']

#     cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
#     output_path = f'fedavg_{alpha}_200clients.txt'
#     cmd_str = " ".join(cmd) + f" > {output_path}"
#     # Print and execute
#     print(f"Running setup: fedavg+{alpha}", flush=True)
#     subprocess.run(cmd_str, shell=True, check=True)

BASE_PARAMS = ['federated_method=fedprox',
    'federated_params.communication_rounds=125',
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=False",]

# for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):

for alpha in alphas:

    params = [ f"dirichlet_alpha={alpha}",] 
    # f'federated_method.epoch_method={e_m}',
    # f'federated_method.iter_method={i_m}',
    # 'federated_method.iterations=3']

    cmd = ["nohup", "python", "train.py"] + BASE_PARAMS + params
    output_path = f'fedprox_{alpha}_200clients.txt'
    cmd_str = " ".join(cmd) + f" > {output_path}"
    # Print and execute
    print(f"Running setup: fedprox+{alpha}", flush=True)
    subprocess.run(cmd_str, shell=True, check=True)
