import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from utils.data_distributions import set_hetero_split

df = pd.read_csv("./cifar10/image_data/cifar10_hetero_map_file.csv")

set_hetero_split(
    df=df,
    name="cifar10",
    target_dir="cifar10",
    amount_of_clients=10,
    head_classes=4,
    head_clients=4,
    random_state=42
)

for i in range(1,11):
    print(f"\nClient {i}, total len = {len(df[df['client']==i])}")
    for j in range(10):
        print(len(df[df["client"] == i][df["target"] == j]), end=" ")
print()