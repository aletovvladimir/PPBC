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

for i in range(1):
    model = resnet18(10).to("cuda")
    model.load_state_dict(torch.load(f"models/client{i}.pt"))
    for _, (x, y) in dataset:
        x = x[0].to("cuda")
        y2 = model(x)
        if y2.isnan().any():
            y = x
            for idx, child in enumerate(list(model.children())):
                y = child(y)
                if y.isnan().any():
                    print(idx, child)
                    break
            break                    
            