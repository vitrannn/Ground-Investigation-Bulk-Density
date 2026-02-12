import torch
import argparse
import numpy as np
from model import Net
from utils import *


parser = argparse.ArgumentParser(description="Model testing")
parser.add_argument("--dir_weights", default="weights/weight.pth", help="The directory of weights")
parser.add_argument("--dir_dataset", default="dataset/Hole Deviation EDA/test.csv", help="The directory of dataset")
parser.add_argument("--dt", choices=["EDA", "GPR"], help="The test set of which dataset?")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net(2)
model.load_state_dict(torch.load('weights/weight.pth'))
model.eval()
model.to(device)

dataset = []
with open(args.dir_dataset, encoding='utf-8') as f:
    data = f.readlines()
    for d in data:
        d_ = d.strip('\n').split(',')
        dataset.append(d_)

if args.dt == "EDA":
    ds = np.array(dataset, dtype=np.float32)[:, [2, 4]]
    labels = np.array(dataset, dtype=np.float32)[:, 8]
elif args.dt == "GPR":
    ds = np.array(dataset, dtype=np.float32)[:, [2, 3]]
    labels = np.array(dataset, dtype=np.float32)[:, 6]
else:
    raise ValueError("Please choose the dataset from EDA and GPR.")

x_min = np.min(ds, axis=0)
x_max = np.max(ds, axis=0)
denom = x_max - x_min
x_min = np.tile(x_min, (ds.shape[0], 1))
denom = np.tile(denom, (ds.shape[0], 1))
data_norm = (ds - x_min) / denom
dataset_ = data_norm.tolist()
preds = []
lbs = []

for i in range(len(dataset_)):
    d = torch.tensor(dataset_[i], dtype=torch.float32).to(device)
    l = torch.tensor(labels[i], dtype=torch.float32).to(device)
    d = d.unsqueeze(dim=0)
    output = model(d).squeeze()
    pred_ = output.detach().cpu().numpy()

    preds.append(pred_)
    lbs.append(l)

r = cal_r(preds, lbs)
r2 = cal_r2(preds, lbs)
aape = cal_aape(preds, lbs)
preds = np.array(preds)
lbs = np.array(lbs)
rmse = np.sqrt((np.square(lbs - preds)).mean(axis=None))

print('r is {}, r2 is {}, aape is {}, rmse is {}'.format(r, r2, aape, rmse))

plot_(np.arange(len(lbs)) + 1, preds, lbs, 'test')


