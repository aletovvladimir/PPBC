import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters

#compression = 'top10'
distr = 'pathology'
method = "pp"
k = 1
t = 0.15

match method:
    case "spp":
        num_rounds = 100
        error_feedback = "EF21"
    case "pp":
        num_rounds = 100
        error_feedback = "EF21"
    case "fedavg":
        num_rounds = 300
        error_feedback = "EF"
    case "fedprox":
        num_rounds = 300
        error_feedback = "EF"
    case "scaffold":
        num_rounds = 300
        error_feedback = "EF"
    case "s-dane":
        num_rounds = 300
        error_feedback = "EF"
        # round_epochs = 10 ??
    case _: 
        num_rounds = 100
        error_feedback = "EF"

EPOCH_METHODS = ['angle'] #'bant', 'gradient_norm', 'loss', 
ITER_METHODS = ['random'] * 4

if False:
    for i in range(4):
        EPOCH_METHODS[i] += '_softmax'

for e_m, i_m in zip(EPOCH_METHODS, ITER_METHODS):
    for compression in ['top10']: # 'top1', 'top5', 
        for rep in [1]:
            params = [
            f'federated_method.epoch_method={e_m}',
            f'federated_method.iter_method={i_m}',
            f'federated_method.epoch_k={k}',
            f'federated_method.iter_k={k}',
            f'federated_method.method={method}'
            ]
            BASE_PARAMS = ['federated_method=ppbc',
                f'federated_params.communication_rounds={num_rounds}',
                f'training_params.compressor={compression}',
                f'training_params.error_feedback={error_feedback}'
            ]

            cmd = ["nohup", "python", "train.py"] + params + BASE_PARAMS
            if "pp" in method:
                output_path = f"log/{distr}/{compression}/{method}/{e_m}_top{k}_t{t}_rep{rep}.txt"
            else:
                output_path = f"log/{distr}/{compression}/{method}/{e_m}_top1_rep{rep}.txt"
            cmd_str = " ".join(cmd) + f" > {output_path}"
            # Print and execute
            print(f"Running setup: {e_m}+{i_m}", flush=True)
            print(f"Output: {output_path}", flush=True)
            subprocess.run(cmd_str, shell=True, check=True)
