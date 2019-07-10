# Project Description

Irish electricity consumption forecasting for the next 7-days using XGBoost.

**Data**

Data contains electricity consumption measurements for 15-min intervals from March, 2011 to February 2013.

Source : [data.gov.ie](https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746)

**Requirements**
* xgboost
* scikit-learn
* matplotlib



**Running Example**

```
$ cd tasks/time-series/time-series-forecasting/XGB/src/

$ mlflow run . -P min_child_weight=[0.8,1.2] -P colsample_bytree=[0.9] -P max_depth=[3,4] -P n_estimators=[500,1000] --no-conda
```
