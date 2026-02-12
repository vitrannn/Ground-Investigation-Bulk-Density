import numpy as np
import pdb
import argparse


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


# dt: EDA or GPR? ts: three sigma or not?
def split(split_r, dt, ts):
    train_data = []
    val_data = []
    test_data = []
    vt = []
    datas = []

    if dt == "EDA":
        with open('./dataset/Hole Deviation EDA/well_log.csv', encoding='utf-8') as f:
            data = f.readlines()[1:]
            for d in data:
                d_ = d.strip('\n').split(',')
                datas.append(d_)

        datas = np.array(datas, dtype=np.float32)
        if ts:
            ds = rmabnormal(datas, 2)
            ds = rmabnormal(ds, 4)
        else:
            ds = datas
        data = ds.tolist()

    elif dt == "GPR":
        with open('./dataset/geomechanical properties/Geomechanical_Properties_of_Rock.csv', encoding='utf-8') as f:
            data = f.readlines()[1:]
            for d in data:
                d_ = d.strip('\n').split(',')
                datas.append(d_)

        datas = np.array(datas, dtype=np.float32)
        if ts:
            ds = rmabnormal(datas, 2)
            ds = rmabnormal(ds, 3)
        else:
            ds = datas
        data = ds.tolist()

    else:
        raise ValueError("Please choose the dataset from EDA and GPR.")

    # with open('./Hole Deviation EDA/filter.csv', 'w', encoding='utf-8') as file:
    #     for d in train_data:
    #         ls = [str(x) for x in d]
    #         file.write(','.join(ls) + '\n')

    data_len = len(data)
    train_data_len = int(float(split_r[0]) * data_len)
    train_seed = np.random.choice(range(data_len), size=train_data_len, replace=False)
    for i in range(data_len):
        if i in train_seed:
            train_data.append(data[i])
        else:
            vt.append(data[i])

    val_data_len = int(float(split_r[1]) * data_len)
    val_seed = np.random.choice(range(len(vt)), size=val_data_len, replace=False)
    for j in range(len(vt)):
        if j in val_seed:
            val_data.append(vt[j])
        else:
            test_data.append(vt[j])

    if dt == "EDA":
        with open('./dataset/Hole Deviation EDA/train.csv', 'w', encoding='utf-8') as file:
            for d in train_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

        with open('./dataset/Hole Deviation EDA/val.csv', 'w', encoding='utf-8') as file:
            for d in val_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

        with open('./dataset/Hole Deviation EDA/test.csv', 'w', encoding='utf-8') as file:
            for d in test_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

    elif dt == "GPR":
        with open('./dataset/geomechanical properties/train.csv', 'w', encoding='utf-8') as file:
            for d in train_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

        with open('./dataset/geomechanical properties/val.csv', 'w', encoding='utf-8') as file:
            for d in val_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

        with open('./dataset/geomechanical properties/test.csv', 'w', encoding='utf-8') as file:
            for d in test_data:
                ls = [str(x) for x in d]
                file.write(','.join(ls) + '\n')

    else:
        raise ValueError("Please choose the dataset from EDA and GPR.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset split")
    parser.add_argument("--split_ratio", nargs=3, default=[0.8, 0.2, 0.0], help="train:val:test")
    parser.add_argument("--dt", default="EDA", help="EDA or GPR?")
    parser.add_argument("--ts", type=bool, help="Three sigma or not?")
    args = parser.parse_args()

    split(args.split_ratio, args.dt, args.ts)



