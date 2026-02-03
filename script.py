import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters

compression = 'top1'
distr = 'pathology'
num_rounds = 300

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
    params = [
    f'federated_method.epoch_method={e_m}',
    f'federated_method.iter_method={i_m}',
    #'federated_method.iterations=1'
    ]

    cmd = ["nohup", "python", "train.py"] + params + BASE_PARAMS
    #output_path = f"logs/EF21/{distr}/{compression}/ppbc/{e_m if not ('softmax' in e_m) else e_m[:-8]}/top3_t0.15.txt"
    output_path = f"logs/EF21/{distr}/{compression}/scaffold/test_noEF.txt"
    cmd_str = " ".join(cmd) + f" > {output_path}"
    # Print and execute
    print(f"Running setup: {e_m}+{i_m}", flush=True)
    print(f"Output: {output_path}", flush=True)
    subprocess.run(cmd_str, shell=True, check=True)
