import subprocess
import os
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Configuration parameters

BASE_PARAMS = ['federated_method=fedavg',
    'federated_params.communication_rounds=30',
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=True",
]
NUM_CLS = [120]
for num in NUM_CLS:
    res = [12, 14]
    # 1. Скачивание — ждём завершения
    cmd = f'python utils/cifar_download.py --amount_of_clients {num}'
    subprocess.run(cmd, shell=True, check=True)

    for re in res:

        params = [f'federated_params.amount_of_clients={num}',
                f'federated_params.round_epochs={re}']

        # 2. Обучение — ждём завершения перед следующей итерацией
        cmd = ["python", "train.py"] + BASE_PARAMS + params
        output_path = f'fedavg_{num}_{re}le.txt'
        with open(output_path, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)
