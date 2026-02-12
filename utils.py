import numpy as np
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
import torch
import random


def train(model, dataset, criterion, optimizer, device):
    model.train()
    mses = []
    outputs = []
    labels = []

    for data in dataset:
        d = data['data'].to(device)
        l = data['label'].to(device)

        out = model(d).squeeze()
        loss = criterion(out, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ls = loss.detach().cpu().numpy()
        output = out.detach().cpu().numpy()
        label = l.detach().cpu().numpy()
        mses.append(ls)
        outputs.extend(output)
        labels.extend(label)

    mse = np.mean(np.array(mses))
    return mse, outputs, labels


def val(model, dataset, criterion, device):
    model.eval()
    mses = []
    outputs = []
    labels = []

    for data in dataset:
        d = data['data'].to(device)
        l = data['label'].to(device)

        out = model(d).squeeze()
        loss = criterion(out, l)

        ls = loss.detach().cpu().numpy()
        output = out.detach().cpu().numpy()
        label = l.detach().cpu().numpy()
        mses.append(ls)
        outputs.extend(output)
        labels.extend(label)

    mse = np.mean(np.array(mses))
    return mse, outputs, labels


def plot(x, y1, y2, name):
    plt.title(name)
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.plot(x, y1, color='r', label='train')
    plt.plot(x, y2, color='b', label='val')
    plt.legend()
    pth = './results/{}.jpg'.format(name)
    plt.savefig(pth)
    plt.close()


def plot_(x, y1, y2, name):
    plt.figure(figsize=(36, 4))
    plt.title(name)
    plt.xlabel('samples')
    plt.ylabel('Values')
    plt.plot(x, y1, color='r', alpha=0.5, label='prediction')
    plt.plot(x, y2, color='b', alpha=0.5, label='label')
    plt.legend()
    pth = './results/{}.jpg'.format(name)
    plt.savefig(pth)
    plt.close()


# def plot_(x, y1, y2, name):
#     plt.title(name)
#     plt.xlabel('samples')
#     plt.ylabel(name)
#     plt.plot(x, y1, color='r', label='prediction')
#     plt.plot(x, y2, color='b', label='label')
#     plt.legend()
#     pth = './results/{}.jpg'.format(name)
#     plt.savefig(pth)
#     plt.close()


def cal_r2(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    a = (pred - label) ** 2
    b = (label - np.mean(label)) ** 2
    r2 = 1 - np.sum(a)/np.sum(b)

    return r2


def vaf(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    a = np.var(label - pred)
    b = np.var(label)
    vaf_ = 1 - a/b

    return vaf_


def cal_r(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    n = len(label)
    a = n * np.sum(pred * label) - np.sum(pred) * np.sum(label)
    b = np.sqrt(n * np.sum(pred ** 2) - (np.sum(pred) ** 2)) * np.sqrt(n * np.sum(label ** 2) - (np.sum(label) ** 2))
    r = a/b

    return r


def cal_aape(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    idx = np.array(np.where(label != 0))[0]
    pred = pred[idx]
    label = label[idx]
    n = len(label)
    aape = np.sum(np.abs((label - pred) / label)) / n

    return aape


def my_loss(output, target):
    loss = torch.mean(torch.sqrt(torch.abs(output - target)))
    return loss


def crossvalidation(lens, k):
    idx = np.arange(lens)
    random.shuffle(idx)
    lists = []
    data_num = lens // k
    for i in range(k):
        data_pairs = []
        data_val = idx[i*data_num: (i+1)*data_num]
        data_tr = [x for x in idx if x not in data_val]
        data_val.tolist()
        data_pairs.append(data_tr)
        data_pairs.append(data_val)
        lists.append(data_pairs)
    return lists


def ci(ds):
    z = 1.96
    datas = np.array(ds, dtype=np.float32)
    average = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    n_r = np.sqrt(datas.shape[0])
    intervals = z * (std / n_r)
    a = round(average, 4)
    b = round(intervals, 4)
    print(f"Average is {a}, and CI is {b}")

