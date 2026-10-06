import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import open_dict
from .image_data_utils import (
    ImageDataset,
    get_image_dataset_params,
)


def prepare_df_for_federated_training(
    cfg: dict,
    directories_key: str,
):
    df = read_dataframe_from_cfg(cfg, directories_key)

    df["client"] = df["client"].apply(lambda x: x - 1)
    df.reset_index(drop=True, inplace=True)

    num_classes = define_number_of_classes(df)

    with open_dict(cfg):
        cfg.training_params.num_classes = num_classes

    print("Preprocess successfull\n")
    return df, cfg


def read_dataframe_from_cfg(
    cfg,
    directories_key="train_directories",
    mode="dataset",
):
    df = pd.DataFrame()
    for directories in cfg[mode]["data_sources"][directories_key]:
        df = pd.concat([df, pd.read_csv(directories, low_memory=False)])
    return df


def get_dataset_loader(
    df: pd.DataFrame,
    cfg,
    drop_last=True,
    mode="train",
    transforms=None,
):
    image_size, mean, std = get_image_dataset_params(cfg, df)
    dataset = ImageDataset(df, mode, image_size, mean, std)

    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def get_stratified_subsample(df, num_samples, random_state):
    """Create a subDataFrame with `num_samples` and stratified label distribution

    Args:
        df (pd.DataFrame): origin DataFrame
        num_samples (_type_): number of samples in subDataFrame

    return: sub_df (pd.DataFrame): sub DataFrame
    """
    sub_df = pd.DataFrame()
    for target in list(df.target.value_counts().keys()):
        tmp = df[df["target"] == target]
        weight = len(tmp) / len(df)
        amount = int(weight * num_samples)
        sub_df = pd.concat(
            [
                sub_df,
                tmp.sample(
                    n=amount,
                    random_state=random_state,
                ),
            ]
        )
    # Remove all rows from df, that are now in sub_df
    df = df[~df["fpath"].isin(list(sub_df["fpath"]))]
    return df, sub_df


def set_up_base_dir(cfg: dict):
    """Set up base directory to handle 'home'/'space' location

    Args:
        cfg (DictConfig): Programm configuration
    """
    # We have four types of directories: client train, server test, filter, server trust
    modes = [
        "dataset",
        "server_test",
        "trust_df",
    ]
    for mode in modes:
        # Inside each type, we have different subtypes (train_directories/valid_directories)
        for type in cfg[mode]["data_sources"]:
            # Each subtype is a list of map_files
            for i in range(len(cfg[mode]["data_sources"][type])):
                path_without_base = "/".join(
                    cfg[mode]["data_sources"][type][i].split("/")[2:]
                )
                fpath = os.path.join("/", cfg.base_dir, path_without_base)
                cfg[mode]["data_sources"][type][i] = fpath
    return cfg


def define_number_of_classes(df):
    if isinstance(df.iloc[0]["target"], list) and len(df.iloc[0]["target"]) > 1:
        return len(df.iloc[0]["target"])  # multilabel case
    else:
        return pd.Series(
            np.concatenate(
                df["target"].apply(lambda x: x if isinstance(x, list) else [x]).values
            )
        ).nunique()
    
def print_df_distribution(df, num_classes, num_clients, pathology_names=None):
    is_multilabel = isinstance(df["target"].iloc[0], list)
    if is_multilabel:
        df["class_target"] = df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    print(f"Total usage data: {len(df[df['client'] != -1])}")

    client_groups = df.groupby("client")
    valid_clients = set(df["client"].unique())

    for cl in range(num_clients):
        if cl not in valid_clients:
            print(f"Client {cl:>2} | No data")
            continue

        client_data = client_groups.get_group(cl)
        if is_multilabel:
            distr = (
                client_data["class_target"]
                .explode()
                .value_counts()
                .reindex(range(-1, num_classes), fill_value=0)
                .tolist()
            )
        else:
            distr = (
                client_data["target"]
                .value_counts()
                .reindex(range(num_classes), fill_value=0)
                .tolist()
            )
        x_str = " ".join(f"{num:>5}" for num in distr)
        if cl == 0 and is_multilabel:
            pathology_str = " " * 11 + "  ".join(["Other"] + pathology_names)
            print(pathology_str)
        print(f"Client {cl:>3} | {x_str} | {len(client_data):>5}")

