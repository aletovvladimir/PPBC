import cv2
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_image(image_path: str, windowing: bool = False):  # move somewhere else
    """
    Load an image from a file using OpenCV or DICOM format.

    Parameters
    ----------
    image_path:
        The path to the image file.
    windowing:
        The flag to apply windowing to DICOM images.

    Returns
    -------
    numpy.ndarray:
        The loaded image as a NumPy array.
        If the image is in DICOM format, it is normalized with values in the range [0, 1]
        and converted to RGB and normalized.
        If the image is in a common image format (e.g., JPEG, PNG), it is loaded and
        color channels are rearranged to RGB.

    Note
    ----
    This function supports loading both standard image formats and DICOM medical
    images. For DICOM images, it assumes that the pixel data is in Hounsfield units
    and normalizes it to the [0, 1] range.
    """

    if image_path.endswith(".dcm") or image_path.endswith(".dicom"):
        raise NotImplementedError("DICOM images are not supported yet.")
    else:
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0
    return image


def get_image_dataset_params(cfg, df):
    # print(cfg.dataset.data_sources.train_directories[0])
    if "cifar10" in cfg.dataset.data_sources.train_directories[0]:
        image_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif "food101" in cfg.dataset.data_sources.train_directories[0]:
        image_size = 224
        mean = (0.5493, 0.4451, 0.3435)
        std = (0.2731, 0.2759, 0.2799)
    else:
        raise NotImplementedError
    return image_size, mean, std


class ImageDataset(Dataset):
    def __init__(self, df, transform_mode, image_size, mean, std):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = self.set_up_transform(transform_mode)

    def __len__(self):
        return len(self.df)

    def set_up_transform(self, mode):
        assert mode in [
            "train",
            "valid",
            "test",
        ], f"ImageDataset works in ['train', 'valid', 'test'] mode, you set {mode}"
        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        return transform

    def __getitem__(self, index):
        image = Image.open(self.df["fpath"][index])
        image = self.transform(image)
        label = self.df["target"][index]
        return index, ([image], label)


def calculate_image_data_metrics(fin_targets, results, verbose=False):
    df = pd.DataFrame(
        columns=["value"],
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "f1-score",
        ],
    )
    df.loc["Accuracy", "value"] = accuracy_score(fin_targets, results)
    df.loc["Precision", "value"] = precision_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["Recall", "value"] = recall_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["f1-score", "value"] = f1_score(
        fin_targets, results, average="macro", zero_division=0
    )
    if verbose:
        print(df)
    return df
