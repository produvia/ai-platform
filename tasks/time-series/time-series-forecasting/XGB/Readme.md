# Project Description

Irish electricity consumption forecasting for the next 7-days using XGBoost.

**Data**

Data contains electricty consumption measurements for 15-min intervals from March, 2011 to February 2013.

Source : [data.gov.ie](https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746)

To download: https://data.smartdublin.ie/dataset/10130831-d2a5-4f4f-b56b-03d8e2cc56c8/resource/6091c604-8c94-4b44-ac52-c1694e83d746/download/dccelectricitycivicsblocks34p20130221-1840.csv

**Requirements**
* xgboost



**Example run**

Usage: python main.py <source_url> <output_file> <predicted_date>

Example: python main.py url.csv ../output.csv 2013-01-07

