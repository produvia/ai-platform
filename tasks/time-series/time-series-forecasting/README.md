# Time Series Forecasting

## Weathernet

Weathernet is a simple LSTM time series forecasting network to predict the temperature of the next day, based on the previous 32 days. Currently it supports only max temperature but can be easily extended to other parameters (e.g. pressure, humidity, etc.)

## Data

Data is from the German Weather Service which provides historical daily climate data for 78 weather stations in Germany. The data is freely available on its website (https://www.dwd.de/DE/leistungen/klimadatendeutschland/klarchivtagmonat.html) - German only. More city data can be downloaded from the website:
![](DWD/DWD_Data_Download.jpg)
Preprocessing is required for data to be used.
This example is using the data from the Hamburg station (1951 until 2018)

## LSTM Network

The LSTM network is a keras functional model and has 4 Layers (1 Input, 2 LSTM and 1 Dense) and currently supports univariate time-series prediction but could easily be extended to multivariate time-series.


Layer (type)                 Output Shape              Param #

seq_input (InputLayer)       (1, 32, 1)                0

cu_dnnlstm_1 (CuDNNLSTM)     (1, 32, 100)              41200

cu_dnnlstm_2 (CuDNNLSTM)     (1, 100)                  80800

output_1 (Dense)             (1, 1)                    101

Total params: 122,101
Trainable params: 122,101
Non-trainable params: 0


## Usage

Project is using MLflow and has two entry-points.
Tensorflow backend is used for training - duration 15s/epoch on GTX1080

### Training only
```bash
mlflow run . -e train_weathernet -P city=CITY
```
### Training and/or predicting
```bash
mlflow run . -e main -P city=CITY
```
Currently as an example Hamburg is supported as CITY.

## MLFlow

Added logging of mlflow metrics and parameters as well as artifacts (loss curves, predictions vs. ground truth)
Start MLflow UI from the command line after training/predicting with:
```bash
mlflow ui
```
In browser enter:
```bash
localhost:5000
```

## Todo
- Create multistep workflow
- Finetune model to catch better on outliers in time-series
