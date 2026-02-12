import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt


def rmabnormal(ds, col):

    datachosen = ds[:, col]
    data = np.array(datachosen)
    d_mean = np.mean(data)
    d_std = np.std(data)
    d_min = d_mean - 3 * d_std
    d_max = d_mean + 3 * d_std

    d_min = np.tile(d_min, data.shape)
    d_max = np.tile(d_max, data.shape)

    sample = (data < d_max) * (data > d_min)
    idx = np.array(np.where(sample == 1))[0]
    data_afterfilter = ds[idx, :]

    return data_afterfilter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draw the histogram of the chosen dataset")
    parser.add_argument("--dt", choices=["EDA", "GPR"], help="The test set of which dataset?")
    parser.add_argument("--bin_num", type=int, default=20, help="The granularity of bin")
    parser.add_argument("--dir_dataset", default="dataset/geomechanical properties/stratified.csv",
                        help="Directory of the dataset")
    parser.add_argument("--ts", type=bool, help="Three sigma applied or not")
    args = parser.parse_args()

    datas = []
    with open(args.dir_dataset, encoding='utf-8') as f:
        data = f.readlines()[1:]
        for d in data:
            d_ = d.strip('\n').split(',')
            datas.append(d_)
    data = np.array(datas, dtype=np.float32)

    if args.ts:
        if args.dt == "GPR":
            datafilter = rmabnormal(data, 2)
            datafilter = rmabnormal(datafilter, 3)
        elif args.dt == "EDA":
            datafilter = rmabnormal(data, 2)
            datafilter = rmabnormal(datafilter, 4)
        else:
            raise ValueError("Please choose the dataset from EDA and GPR.")
    else:
        datafilter = data

    if args.dt == "GPR":
        resistivity = datafilter[:, 2]
        gamma = datafilter[:, 3]
        density = datafilter[:, 6]
    elif args.dt == "EDA":
        resistivity = datafilter[:, 4]
        gamma = datafilter[:, 2]
        density = datafilter[:, 8]
    else:
        raise ValueError("Please choose the dataset from EDA and GPR.")

    plt.figure(1)
    plt.hist(resistivity, bins=args.bin_num)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('resistivity')

    plt.figure(2)
    plt.hist(gamma, bins=args.bin_num)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('gamma')

    plt.figure(3)
    plt.hist(density, bins=args.bin_num)
    plt.xlabel('value')
    plt.ylabel('density')
    plt.title('density')

    plt.show()

