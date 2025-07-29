import torch
from torch.utils.data import Dataset
import numpy as np

class BinTokenDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.df = df
        self.T = sequence_length
        self.tokens = self._load_all_tokens()
        self.num_samples = (len(self.tokens) - 1) // self.T

    def _load_all_tokens(self):
        all_tokens = []
        for path in self.df["fpath"]:
            with open(path, "rb") as f:
                header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
                assert header[0] == 20240520
                ntok = header[2]
                toks = np.frombuffer(f.read(), dtype=np.uint16)
                assert len(toks) == ntok
                all_tokens.append(toks)
        return np.concatenate(all_tokens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.T
        x = self.tokens[start:start+self.T]
        y = self.tokens[start+1:start+self.T+1]
        return idx, (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
