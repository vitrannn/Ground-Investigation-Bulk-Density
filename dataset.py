import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Drill(Dataset):
    def __init__(self, dataset, data):
        self.data = data
        if data == "EDA":
            self.label = dataset[:, 8]

        elif data == "GPR":
            self.label = dataset[:, 6]

        else:
            raise ValueError("Please choose the dataset from EDA and GPR.")

        ds = np.array(dataset, dtype=np.float32)
        x_min = np.min(ds, axis=0)
        x_max = np.max(ds, axis=0)
        denom = x_max - x_min
        x_min = np.tile(x_min, (ds.shape[0], 1))
        denom = np.tile(denom, (ds.shape[0], 1))
        data_norm = (ds - x_min) / denom
        self.dataset = data_norm.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        d = self.dataset[item]
        if self.data == "GPR":
            input_data = np.array([d[2], d[3]], dtype=np.float32)
        elif self.data == "EDA":
            input_data = np.array([d[2], d[4]], dtype=np.float32)
        else:
            raise ValueError("Please choose the dataset from EDA and GPR.")
        lb = self.label[item]
        data = torch.tensor(input_data).type(torch.float32)
        label = torch.tensor(lb)

        return {'data': data, 'label': label}




