# Bulk density prediction
Predicting Bulk Density by Logging-while-Drilling Data based on ANN Model

## Environment
By running:
```bash
pip install -r requirements.txt
```

## Steps for training and evaluating
1. (Optional) Split the data set by running
```bash
python ./dataset/data_split.py --split_ratio 0.8 0.2 0.0 --dt "EDA" --ts True
```
This is to split the chosen dataset according to the `--split_ratio`, which indicates the training set, validation set and test set respectively. `--ts` refers to whether apply three-sigma rule to filter the data or not. Drop this flag if False is wanted.
The output will be saved as csv file in the directory of the corresponding dataset.

2. (Optional) Draw the histogram of the chosen dataset
```bash
python histo.py --dt "EDA" --bin_num 20 --dir_dataset "dataset/geomechanical properties/stratified.csv" --ts True
```
To draw the histogram of any chosen dataset, where the `--bin_num` refers to the number of bins when drawing the histogram, and the `--dir_dataset` refers to the directory of the dataset.
The results will be shown as histogram pictures, including gamma ray, resistivity and density.

3. Training the model with certain dataset by cross validation
```bash
python main.py --bs 16 --lr 1e-4 --weight_decay 1e-4 --eps 5000 --k 5 --dir './dataset/geomechanical properties/stratified.csv' --dt "GPR"
```
`--k` refers to the k-fold. The weights will be saved in `/weights` folder
The metrics curves will be saved for the running on each fold in the `/results` folder, and named in the form of `{metric}_{fold}`, where metric contains rmse, aape, r2 and r. 
Four metrics for each epoch will also be printed, and the best ones will be printed in the last.

4. Test the weights on the test set
```bash
python test.py --dir_weights "weights/weight.pth" --dir_dataset "dataset/Hole Deviation EDA/test.csv" --dt "EDA"
```
Four metrics will be printed, and the comparison between the labels and prediction will be saved as picture in `/results` folder.

5. Conduct stratified sampling from GPR dataset
```bash
python stratifiedsampling.py --dir "./dataset/geomechanical properties/train.csv" --sample_num 900 --bin_num 20
```
`--sample_num` defines how many samples are wanted from the original dataset, and `--bin_num` refers to the granularity when conducting sampling.
The samples will be saved as `/dataset/geomechanical properties/stratified.csv`.

6. To calculate the confidence interval of an input list, call function `ci` in `utils.py`.

