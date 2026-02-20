import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters

compression = 'top5'
distr = 'pathology'
method = "ppbc"

match method:
    case "sppbc":
        num_rounds = 100
    case "ppbc":
        num_rounds = 100
    case "fedavg":
        num_rounds = 300
    case "fedprox":
        num_rounds = 300
    case "scaffold":
        num_rounds = 300
    case _: num_rounds = 100
num_rounds = 31

BASE_PARAMS = ['federated_method=ppbc',
    f'federated_params.communication_rounds={num_rounds}',
    f'training_params.compressor={compression}'
]

EPOCH_METHODS = ['bant', 'gradient_norm', 'loss', 'angle']#
ITER_METHODS = ['random'] * 4
EPOCH_METHODS = ['loss']
ITER_METHODS = ['random']

if False:
    for i in range(4):
        EPOCH_METHODS[i] += '_softmax'

for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):
    for t in [0.05, 0.1, 0.15, 0.2, 0.25]:
        params = [
        f'federated_method.epoch_method={e_m}',
        f'federated_method.iter_method={i_m}',
        f'federated_method.epoch_k={8}',
        f'federated_method.iter_k={8}',
        f'federated_method.method={method}'
        ]

        cmd = ["nohup", "python", "train.py"] + params + BASE_PARAMS
        if "ppbc" in method:
            output_path = f"logs/EF21/{distr}/{compression}/ppbc/{e_m if not ('softmax' in e_m) else e_m[:-8]}/TEST4_top8_t{t}_.txt"
        else:
            output_path = f"logs/EF21/{distr}/{compression}/{method}/{e_m}_top1.txt"
        cmd_str = " ".join(cmd) + f" > {output_path}"
        # Print and execute
        print(f"Running setup: {e_m}+{i_m}", flush=True)
        print(f"Output: {output_path}", flush=True)
        subprocess.run(cmd_str, shell=True, check=True)
