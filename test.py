import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import torch
from utils.model_utils import resnet18
from utils.data_utils import get_dataset_loader
import yaml
from omegaconf import OmegaConf

with open("configs/config.yaml", "r") as f:
    cfg = OmegaConf.load(f)

df = pd.read_csv("/home/DBystrov/EF25_NIPS/cifar10/image_data/cifar10_pathology_map_file.csv")
dataset = get_dataset_loader(df, cfg)

for i in range(10):
    model = resnet18(10).to("cuda")
    model.load_state_dict(torch.load(f"test_client{i}.pt"))
    print(i)
    for _, (x, y) in dataset:
        y2 = model(x[0].to("cuda"))
        y3 = torch.softmax(y2, dim=1)
        if y2.isnan().any():
            print(_, "2")
        if y3.isnan().any():
            print(_, "3")