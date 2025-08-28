import torch
from torch.utils.data import Dataset
import numpy as np


FEATURES = ["rv_10","z_30","rsi_14","macd","macd_sig","sma_50","sma_200","vol_spike","gbm_mu","gbm_sigma"]
TARGET = "r1" # next‑day log‑return


class WindowedDS(Dataset):
    def __init__(self, df, lookback=30):
        self.X, self.y = [], []
        arr = df[FEATURES + [TARGET]].values.astype(np.float32)
        for i in range(lookback, len(arr)-1):
            x = arr[i-lookback:i, :-1]
            y = arr[i+1, -1] # predict t+1 return distribution
            self.X.append(x)
            self.y.append(y)
        self.X = torch.tensor(np.stack(self.X))
        self.y = torch.tensor(np.array(self.y))


    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]