import random
import numpy as np
import argparse


def stratified_sample(num, bin_, r):
    datas = []
    with open(r, encoding='utf-8') as f:
        data = f.readlines()
        for d in data:
            d_ = d.strip('\n').split(',')
            datas.append(d_)

    datas = np.array(datas, dtype=np.float32)
    data_pairs_sort = datas[np.argsort(datas[:, 6])]
    density = datas[:, 6]
    hist, bins = np.histogram(density, bins=bin_)

    total_num = len(datas)
    sample_num = (num / total_num) * hist
    sample_num = [round(x) for x in sample_num]

    idx = []
    sum_idx = 0
    for i in range(bin_):
        id_ = random.sample(np.arange(sum_idx, sum(hist[:i+1])).tolist(), sample_num[i])
        idx.extend(id_)
        sum_idx = sum_idx + hist[i]

    data_chosen = data_pairs_sort[idx, :]

    with open('./dataset/geomechanical properties/stratified.csv', 'w', encoding='utf-8') as file:
        for d in data_chosen:
            ls = [str(x) for x in d]
            file.write(','.join(ls) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified sampling from GPR dataset")
    parser.add_argument("--dir", help="The directory of the dataset")
    parser.add_argument("--sample_num", type=int, default=900, help="The total amount of samples from the dataset")
    parser.add_argument("--bin_num", type=int, default=20, help="The granularity of bin")
    args = parser.parse_args()

    stratified_sample(args.sample_num, args.bin_num, args.dir)








