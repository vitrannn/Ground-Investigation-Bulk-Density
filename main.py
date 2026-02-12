import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Subset
from pytorchtool import EarlyStopping
from model import Net
from dataset import Drill
from utils import *


parser = argparse.ArgumentParser(description="Model training and validation")
parser.add_argument("--bs", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--eps", type=int, default=5000, help="Epoches")
parser.add_argument("--k", type=int, default=5, help="KFold")
parser.add_argument("--dir", help="The directory of the dataset")
parser.add_argument("--dt", choices=["EDA", "GPR"], help="The dataset used")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# r = r'./dataset/geomechanical properties/stratified.csv'
# r = r'./dataset/Hole Deviation EDA/filter.csv'

dataset = []
with open(args.dir, encoding='utf-8') as f:
    data = f.readlines()
    for d in data:
        d_ = d.strip('\n').split(',')
        dataset.append(d_)

crossval = crossvalidation(len(dataset), args.k)

dataset = np.array(dataset, dtype=np.float32)

cv_rmse_tr = []
cv_aape_tr = []
cv_r2_tr = []
cv_r_tr = []

cv_rmse_val = []
cv_aape_val = []
cv_r2_val = []
cv_r_val = []

eps = []

for i in range(args.k):
    model = Net(2)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=100, verbose=True)

    tv_seed = crossval[i]
    t_seed = tv_seed[0]
    v_seed = tv_seed[1]

    train_data = dataset[t_seed]
    validation_data = dataset[v_seed]

    tr_set = Drill(train_data, args.dt)
    val_set = Drill(validation_data, args.dt)

    tr_loader = DataLoader(dataset=tr_set, batch_size=args.bs, shuffle=True, drop_last=True)
    va_loader = DataLoader(dataset=val_set, batch_size=args.bs, drop_last=True)

    tr_loss = []
    va_loss = []
    rmse_tr = []
    rmse_va = []
    tr_r2 = []
    va_r2 = []
    aape_tr = []
    aape_va = []
    r_tr = []
    r_va = []

    best_va_out = None
    best_va_lb = None
    best_tr_out = None
    best_tr_lb = None

    best_rmse = np.Inf

    ep_ = 0

    for ep in range(args.eps):
        tr_mse, tr_out, tr_lb = train(model, tr_loader, criterion, optimizer, device)

        r2_tr = format(cal_r2(tr_out, tr_lb), '.4f')
        tr_rmse = format(np.sqrt(tr_mse), '.4f')
        tr_aape = format(cal_aape(tr_out, tr_lb), '.4f')
        tr_r = format(cal_r(tr_out, tr_lb), '.4f')

        print('The rmse for training in epoch {} is: {}'.format(ep + 1, tr_rmse))
        print('The r2 for training in epoch {} is: {}'.format(ep + 1, r2_tr))
        print('The aape for training in epoch {} is: {}'.format(ep + 1, tr_aape))
        print('The r for training in epoch {} is: {}'.format(ep + 1, tr_r))

        va_mse, va_out, va_lb = val(model, va_loader, criterion, device)

        r2_va = format(cal_r2(va_out, va_lb), '.4f')
        va_rmse = format(np.sqrt(va_mse), '.4f')
        va_aape = format(cal_aape(va_out, va_lb), '.4f')
        va_r = format(cal_r(va_out, va_lb), '.4f')

        print('The rmse for validating in epoch {} is: {}'.format(ep + 1, va_rmse))
        print('The r2 for validating in epoch {} is: {}'.format(ep + 1, r2_va))
        print('The aape for validating in epoch {} is: {}'.format(ep + 1, va_aape))
        print('The r for validating in epoch {} is: {}'.format(ep + 1, va_r))

        if float(va_rmse) < float(best_rmse):
            best_rmse = va_rmse
            best_va_out = va_out
            best_va_lb = va_lb
            best_tr_out = tr_out
            best_tr_lb = tr_lb
            torch.save(model.state_dict(), f'weights/weight{i}_{ep}.pth')

        tr_loss.append(float(tr_rmse))
        va_loss.append(float(va_rmse))
        tr_r2.append(float(r2_tr))
        va_r2.append(float(r2_va))
        aape_tr.append(float(tr_aape))
        aape_va.append(float(va_aape))
        r_tr.append(float(tr_r))
        r_va.append(float(va_r))
        rmse_tr.append(float(tr_rmse))
        rmse_va.append(float(va_rmse))

        ep_ = ep + 1

        early_stopping(va_mse, model)
        if early_stopping.early_stop:
            break

    eps.append(ep_)

    plot(np.arange(ep_) + 1, tr_loss, va_loss, 'rmse_{}'.format(i))
    plot(np.arange(ep_) + 1, aape_tr, aape_va, 'aape_{}'.format(i))
    plot(np.arange(ep_) + 1, tr_r2, va_r2, 'r2_{}'.format(i))
    plot(np.arange(ep_) + 1, r_tr, r_va, 'r_{}'.format(i))

    va_out = np.array(best_va_out, dtype=np.float32)
    va_lb = np.array(best_va_lb, dtype=np.float32)
    tr_out = np.array(best_tr_out, dtype=np.float32)
    tr_lb = np.array(best_tr_lb, dtype=np.float32)

    plot_(np.arange(len(tr_out)) + 1, tr_out, tr_lb, 'Comparison in training set _ {}'.format(i))
    plot_(np.arange(len(va_out)) + 1, va_out, va_lb, 'Comparison in validation set _ {}'.format(i))

    best_aape_t = min(aape_tr)
    best_aape_e_t = np.argmin(np.array(aape_tr))
    best_r_t = max(r_tr)
    best_r_e_t = np.argmax(np.array(r_tr))
    best_r2_t = max(tr_r2)
    best_r2_e_t = np.argmax(np.array(tr_r2))
    best_rmse_t = min(rmse_tr)
    best_rmse_e_t = np.argmin(np.array(rmse_tr))

    cv_rmse_tr.append(best_rmse_t)
    cv_aape_tr.append(best_aape_t)
    cv_r2_tr.append(best_r2_t)
    cv_r_tr.append(best_r_t)

    best_aape_v = min(aape_va)
    best_aape_e_v = np.argmin(np.array(aape_va))
    best_r_v = max(r_va)
    best_r_e_v = np.argmax(np.array(r_va))
    best_r2_v = max(va_r2)
    best_r2_e_v = np.argmax(np.array(va_r2))
    best_rmse_v = min(rmse_va)
    best_rmse_e_v = np.argmin(np.array(rmse_va))

    cv_rmse_val.append(best_rmse_v)
    cv_aape_val.append(best_aape_v)
    cv_r2_val.append(best_r2_v)
    cv_r_val.append(best_r_v)

best_rmse_t_ = np.average(np.array(cv_rmse_tr))
best_aape_t_ = np.average(np.array(cv_aape_tr))
best_r2_t_ = np.average(np.array(cv_r2_tr))
best_r_t_ = np.average(np.array(cv_r_tr))

best_rmse_v_ = np.average(np.array(cv_rmse_val))
best_aape_v_ = np.average(np.array(cv_aape_val))
best_r2_v_ = np.average(np.array(cv_r2_val))
best_r_v_ = np.average(np.array(cv_r_val))

# print('tr_rmse: {}\ntr_aape: {}\ntr_r2: {}\ntr_r: {}\nval_rmse: {}\nval_aape: {}\nval_r2: {}\nval_r:{}'.
#       format(cv_rmse_tr, cv_aape_tr, cv_r2_tr, cv_r_tr, cv_rmse_val, cv_aape_val, cv_r2_val, cv_r_val))

print('epochs: {}, average is {}'.format(eps, np.average(np.array(eps))))

print('the best rmse in training set is {}.\n'
      'the best aape in training set is {}.\n'
      'the best r2 in training set is {}.\n'
      'the best r in training set is {}.'.format(best_rmse_t_, best_aape_t_, best_r2_t_, best_r_t_))

print('the best rmse in validation set is {}.\n'
      'the best aape in validation set is {}.\n'
      'the best r2 in validation set is {}.\n'
      'the best r in validation set is {}.'.format(best_rmse_v_, best_aape_v_, best_r2_v_, best_r_v_))








