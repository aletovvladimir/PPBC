import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def set_uniform_split(
    df,
    target_dir,
    name,
    amount_of_clients=10,
):
    target_sizes = np.unique(df["target"], return_counts=True)[1]
    clients = []

    for size in target_sizes:
        for_each_client = size // amount_of_clients
        diff = size % amount_of_clients

        client_dist = [[i + 1] * for_each_client for i in range(amount_of_clients)]
        clients.append(np.concatenate(client_dist))
        if diff != 0:
            clients.append([amount_of_clients + 1] * diff) if diff != 0 else 0

    df["client"] = np.concatenate(clients)
    path_to_save = Path(target_dir) / "image_data" / f"{name}_train_map_file.csv"

    df.to_csv(path_to_save, index=False)


def set_pathology_split(
    df, std, name, target_dir="food101", amount_of_clients=10, random_state=RANDOM_SEED
):
    mean = len(df) // amount_of_clients
    rng = np.random.default_rng(seed=random_state)
    amount_for_each_client = [
        rng.integers(mean - int(mean * std), mean + int(mean * std))
        for _ in range(amount_of_clients - 1)
    ]
    amount_for_each_client.append(len(df) - sum(amount_for_each_client))

    for _ in range(100):
        df = df.sample(frac=1).reset_index(drop=True)

    clients = np.concatenate(
        [[i + 1] * amount_for_each_client[i] for i in range(amount_of_clients)]
    )
    df["client"] = clients

    path_to_save = Path(target_dir) / "image_data" / f"{name}_pathology_map_file.csv"
    add_info_path = Path(target_dir) / "image_data" / f"pathology_additional_info"
    df.to_csv(path_to_save, index=False)
    np.save(file=add_info_path, arr=amount_for_each_client)


def flexible_split(
    df, amount_of_clients=10, head_classes=20, head_clients=2, random_state=RANDOM_SEED
):
    rng = np.random.default_rng(random_state)
    num_classes = df["target"].nunique()
    clients = [[] for _ in range(amount_of_clients)]

    for class_id in range(num_classes):
        class_df = df[df["target"] == class_id]
        indices = class_df.sample(frac=1, random_state=random_state).index.tolist()

        if len(indices) < amount_of_clients * 2:
            raise ValueError(
                f"Not enough samples in class {class_id}: minimum {amount_of_clients * 2} is needed"
            )

        for i in range(amount_of_clients):
            clients[i].extend(indices[i * 2 : (i + 1) * 2])

        remaining = indices[amount_of_clients * 2 :]

        if class_id < head_classes:
            for i, idx in enumerate(remaining):
                client_id = i % head_clients
                clients[client_id].append(idx)
        else:
            for i, idx in enumerate(remaining):
                client_id = head_clients + (i % (amount_of_clients - head_clients))
                clients[client_id].append(idx)

    return clients


def assign_clients_to_df(df, clients):
    df = df.copy()
    client_column = pd.Series(index=df.index, dtype=int)

    for client_id, indices in enumerate(clients):
        client_column.loc[indices] = client_id

    df["client"] = client_column.astype(np.int64)
    return df


def set_hetero_split(
    df,
    name,
    target_dir="food101",
    amount_of_clients=10,
    head_classes=20,
    head_clients=2,
    random_state=RANDOM_SEED,
):
    clients = flexible_split(
        df,
        amount_of_clients=amount_of_clients,
        head_classes=head_classes,
        head_clients=head_clients,
        random_state=random_state,
    )
    df = assign_clients_to_df(df, clients)

    path_to_save = Path(target_dir) / "image_data" / f"{name}_hetero_map_file.csv"

    df.to_csv(path_to_save, index=False)
