import os
import tarfile
import requests
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path

from data_distributions import set_uniform_split, set_pathology_split, set_hetero_split

import fire


# Set global seeds for reproducibility

CLIENTS_AMOUNT = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def download_cifar10(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = Path(target_dir) / "cifar-10-python.tar.gz"

    if not tar_path.exists():
        print("Downloading CIFAR-10...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            f.write(response.content)

    # Extract files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    tar_path.unlink()


def process_cifar10(base_dir):
    img_dir = Path(base_dir) / "data"
    img_dir.mkdir(parents=True, exist_ok=True)

    map_file_data_path = Path(base_dir) / "image_data"
    map_file_data_path.mkdir(parents=True, exist_ok=True)

    with open(Path(base_dir) / "cifar-10-batches-py" / "batches.meta", "rb") as f:
        meta = pickle.load(f)
    label_names = meta["label_names"]

    fpath, targets, names = [], [], []
    print("Converting CIFAR-10...")
    for split in ["train", "test"]:
        files = (
            ["data_batch_%d" % i for i in range(1, 6)]
            if split == "train"
            else ["test_batch"]
        )

        for file in files:
            with open(Path(base_dir) / "cifar-10-batches-py" / file, "rb") as f:
                data = pickle.load(f, encoding="bytes")

            images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = data[b"labels"]

            for i, (img, label) in enumerate(zip(images, labels)):
                if split == "train":
                    file_split = file.split("_")[-1]
                else:
                    file_split = split
                filename = f"{label_names[label]}_{file_split}_split_{i:06d}.png"
                path = img_dir / filename
                Image.fromarray(img).save(path)
                fpath.append(path)
                names.append(filename)
                targets.append(label)

        if split == "test":
            df_test = pd.DataFrame({"fpath": fpath, "name": names, "target": targets})

            df_test.to_csv(
                map_file_data_path / "cifar10_test_map_file.csv", index=False
            )
        else:
            df_train = pd.DataFrame({"fpath": fpath, "name": names, "target": targets})

    return df_train


def split_train_and_trust(df, trust_data_size, target_dir):
    targets = np.unique(df["target"])
    trust_df = pd.DataFrame()
    for target in targets:
        sub_df = df[df["target"] == target]

        df_for_trust = sub_df.sample(frac=trust_data_size)
        df = df.drop(df_for_trust.index)

        trust_df = pd.concat([trust_df, df_for_trust], axis=0)

    path_to_save_trust = Path(target_dir) / "image_data" / "cifar10_trust_map_file.csv"
    path_to_save_train = (
        Path(target_dir) / "image_data" / "cifar10_train_no_cls_map_file.csv"
    )

    trust_df.to_csv(path_to_save_trust)
    df.to_csv(path_to_save_train)

    return df


def train_test_trust_splits(target_dir, trust_data_size):
    download_cifar10(target_dir=target_dir)
    df = process_cifar10(base_dir=target_dir)
    print(f"Downloading is done!")
    df = split_train_and_trust(
        df=df, target_dir=target_dir, trust_data_size=trust_data_size
    )

    return df


def prepare_splits(df, target_dir):
    print("Preparing splits for different distributions")

    set_uniform_split(
        df=df, target_dir=target_dir, amount_of_clients=120, name="cifar10"
    )
    print("Uniform split is done")

    set_pathology_split(
        df=df,
        std=0.1,
        target_dir=target_dir,
        amount_of_clients=CLIENTS_AMOUNT,
        random_state=42,
        name="cifar10",
    )
    print("Pathology split is done")

    set_hetero_split(
        df,
        target_dir=target_dir,
        amount_of_clients=CLIENTS_AMOUNT,
        head_classes=30,
        head_clients=4,
        random_state=42,
        name="cifar10",
    )
    print("Hetero split is done")

    print("All splits is done!")


def set_data_configs(target_dir):
    print("Setting paths to .yaml files...")
    config_dir = Path("configs") / "observed_data_params"
    if not config_dir.is_dir():
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

    config_names = [
        "cifar10.yaml",
        "cifar10_pathology.yaml",
        "cifar10_hetero.yaml",
        "cifar10_trust.yaml",
    ]

    if not Path(target_dir).is_absolute():
        curent_run_path = Path.cwd()
        target_dir = curent_run_path / target_dir
    for filename in Path(config_dir).iterdir():
        if filename.name not in config_names:
            continue

        filepath = filename
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})
        distr_info = data.get("distribution_info", {})

        if "test_directories" in data_sources:
            test_map_path = target_dir / "image_data" / "cifar10_test_map_file.csv"
            data_sources["test_directories"] = [str(test_map_path)]

        if "train_directories" in data_sources:
            if filename.name in ["cifar10.yaml"]:
                train_map_name = "cifar10_train_map_file.csv"
            elif filename.name == "cifar10_hetero.yaml":
                train_map_name = "cifar10_hetero_map_file.csv"
            elif filename.name == "cifar10_trust.yaml":
                train_map_name = "cifar10_trust_map_file.csv"
            else:
                train_map_name = "cifar10_pathology_map_file.csv"
                distribution = "pathology_additional_info.npy"
                distr_map_file = target_dir / "image_data" / distribution
                distr_info = distr_map_file
                data["distribution_info"] = [str(distr_info)]

            if train_map_name is not None:
                train_map_path = target_dir / "image_data" / train_map_name
                data_sources["train_directories"] = [str(train_map_path)]

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    print("All steps completed successfully!!!")


def prepare_data(target_dir="cifar10", trust_data_size=0.05):
    df = train_test_trust_splits(target_dir=target_dir, trust_data_size=trust_data_size)
    prepare_splits(df=df, target_dir=target_dir)
    set_data_configs(target_dir=target_dir)


if __name__ == "__main__":
    fire.Fire(prepare_data)
