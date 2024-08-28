# Pattern-oriented Attention Mechanism for Multivariate Time Series Forecasting

## Datasets
This repository contains the code of `PRformer`. Seven datasets are available [here](https://www.kaggle.com/datasets/limpidcloud/datasets-for-multivariate-time-series-forecasting), including `ECL`, `ETTh`, `ETTm`, `QPS`, `Solar`, `Traffic` and `Weather`. They should be firstly unzipped and moved into the `dataset` folder.

Dataset Lists: 
* dataset
  * ETTh.csv
  * ETTm.csv
  * LD2011_2014.csv
  * mpi_roof.csv
  * PeMS.csv
  * QPS.csv
  * solar_Alabama.csv

## Run
You can run `main.py` to reproduce the experiment. Below is an example of running the `Traffic` dataset with `pred_len = 24`.
```
python3 main.py -cuda_id 0 -dataset Traffic -pred_len 24
```
There are seven dataset names: 
```
ECL ETTh ETTm QPS Solar Traffic Weather
```
and their hyperparameters are listed in `configs.json`.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.
