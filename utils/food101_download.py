from pathlib import Path
import pandas as pd
import datasets
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import ImageFile
import numpy as np

from data_distributions import set_uniform_split, set_pathology_split, set_hetero_split

import yaml
import fire

ImageFile.LOAD_TRUNCATED_IMAGES = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def save_example(index, mode, global_path, all_data):
    try:
        item = all_data[index]
        image, label = item["image"], item["label"]
        image = image.convert("RGB")
        filename = f"{index}_{mode}.png"
        image_path = global_path / filename
        image.save(image_path, format="png")
        return image_path, label, filename
    except Exception as e:
        print(f"[Error] image with index {index} is missed: {e}")
        return None


def get_data(target_dir, mode, num_workers):
    all_data = (
        datasets.load_dataset("ethz/food101")[mode]
        if mode == "train"
        else datasets.load_dataset("ethz/food101")["validation"]
    )

    image_global_path = Path(target_dir) / "data"
    image_global_path.mkdir(parents=True, exist_ok=True)

    map_file_data_path = Path(target_dir) / "image_data"
    map_file_data_path.mkdir(parents=True, exist_ok=True)

    fpath, targets, names = [], [], []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        from functools import partial

        save_func = partial(
            save_example, mode=mode, global_path=image_global_path, all_data=all_data
        )
        futures = [executor.submit(save_func, i) for i in range(len(all_data))]

        for future in tqdm(futures, desc="Saving images"):
            res = future.result()
            if res is None:
                continue
            path, label, name = res
            fpath.append(path)
            targets.append(label)
            names.append(name)

    df = pd.DataFrame({"fpath": fpath, "target": targets, "name": names})
    df = df.sort_values(by="target").reset_index(drop=True)

    if mode == "test":
        map_file_data_path = map_file_data_path / "food101_test_map_file.csv"
        df.to_csv(map_file_data_path, index=False)

    return df


def split_train_and_trust(df, trust_data_size, target_dir):
    targets = np.unique(df["target"])
    trust_df = pd.DataFrame()
    for target in targets:
        sub_df = df[df["target"] == target]

        df_for_trust = sub_df.sample(frac=trust_data_size)
        df = df.drop(df_for_trust.index)

        trust_df = pd.concat([df, df_for_trust], axis=0)

    path_to_save_trust = Path(target_dir) / "image_data" / "food101_trust_map_file.csv"
    path_to_save_train = (
        Path(target_dir) / "image_data" / "food101_train_no_cls_map_file.csv"
    )

    trust_df.to_csv(path_to_save_trust)
    df.to_csv(path_to_save_train)

    return df


def train_test_trust_splits(target_dir, num_workers, trust_data_size):
    _ = get_data(target_dir=target_dir, mode="test", num_workers=num_workers)
    df = get_data(target_dir=target_dir, mode="train", num_workers=num_workers)
    print(f"Downloading is done!")
    df = split_train_and_trust(
        df=df, target_dir=target_dir, trust_data_size=trust_data_size
    )

    return df


def prepare_splits(df, target_dir):
    print("Preparing splits for different distributions")

    set_uniform_split(
        df=df, target_dir=target_dir, amount_of_clients=10, name="food101"
    )
    print("Uniform split is done")

    set_pathology_split(
        df=df,
        std=0.1,
        target_dir=target_dir,
        amount_of_clients=10,
        random_state=42,
        name="food101",
    )
    print("Pathology split is done")

    set_hetero_split(
        df,
        target_dir=target_dir,
        amount_of_clients=10,
        head_classes=30,
        head_clients=4,
        random_state=42,
        name="food101",
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
        "food101.yaml",
        "food101_pathology.yaml",
        "food101_hetero.yaml",
        "food101_trust.yaml",
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
            test_map_path = target_dir / "image_data" / "food101_test_map_file.csv"
            data_sources["test_directories"] = [str(test_map_path)]

        if "train_directories" in data_sources:
            if filename.name in ["food101.yaml"]:
                train_map_name = "food101_train_map_file.csv"
            elif filename.name == "food101_hetero.yaml":
                train_map_name = "food101_hetero_map_file.csv"
            elif filename.name == "food101_trust.yaml":
                train_map_name = "food101_trust_map_file.csv"
            else:
                train_map_name = "food101_pathology_map_file.csv"
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


def prepare_data(target_dir="food101", num_workers=8, trust_data_size=0.05):
    df = train_test_trust_splits(
        target_dir=target_dir, num_workers=num_workers, trust_data_size=trust_data_size
    )
    prepare_splits(df=df, target_dir=target_dir)
    set_data_configs(target_dir=target_dir)


if __name__ == "__main__":
    fire.Fire(prepare_data)
