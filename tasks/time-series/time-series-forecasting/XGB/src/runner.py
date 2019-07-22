import pandas as pd
import numpy as np
import io
import requests
from datetime import timedelta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


class ForecastRunner(object):
    def __init__(self, url, output_file, predicted_date, min_child_weight, colsample_bytree, max_depth, n_estimators):
        self.url = url
        self.output_file = output_file
        self.predicted_date = predicted_date
        self.min_child_weight = [float(i) for i in eval(str(min_child_weight))]
        self.colsample_bytree = [float(i) for i in eval(str(colsample_bytree))]
        self.max_depth = [int(i) for i in eval(str(max_depth))]
        self.n_estimators = [int(i) for i in eval(str(n_estimators))]

    def get_input(self):
        s = requests.get(self.url).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')), header=1)
        # sum along 15-min intervals and convert into daily values
        df['Value'] = df.drop(['Date', 'Values'], axis=1).sum(axis=1)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        input_data = df[['Date', 'Value']]
        return input_data

    def save_output(self, test, preds):
        preds = preds.reset_index(drop=True)
        df_test = test.reset_index()[['Date']]
        prediction = df_test.join(preds)
        prediction.to_csv(self.output_file)

    @staticmethod
    def evaluation_metrics(y_true, y_pred):
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mape, rmse, mae, r2

    @staticmethod
    def remove_outliers(data, fill=False, threshold=3.5):
        """
        Median Absolute Deviation (MAD) based outlier detection
        Removes outliers and if selected fills with polynomial  interpolation
        fill: Boolean
        """
        median = np.median(data.values, axis=0)
        diff = np.sum((data.values - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        # scale constant 0.6745
        modified_z_score = 0.6745 * diff / med_abs_deviation
        data[modified_z_score > threshold] = np.nan

        if fill:
            # fill by interpolation
            data = data.interpolate(method='polynomial', order=2)
        data = data.dropna()
        return data

    def prepare_data(self, df, fill=False):
        df = df.set_index('Date')
        # get time features
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Week'] = df.index.week
        df['DOW'] = df.index.weekday

        # encode time features with the mean of the target variable
        yearly_avg = dict(df.groupby('Year')['Value'].mean())
        df['year_avg'] = df['Year'].apply(lambda x: yearly_avg[x])
        monthly_avg = dict(df.groupby('Month')['Value'].mean())
        df['month_avg'] = df['Month'].apply(lambda x: monthly_avg[x])
        weekly_avg = dict(df.groupby('Week')['Value'].mean())
        df['week_avg'] = df['Week'].apply(lambda x: weekly_avg[x])
        dow_avg = dict(df.groupby('DOW')['Value'].mean())
        df['dow_avg'] = df['DOW'].apply(lambda x: dow_avg[x])

        df = df.drop(['Year', 'Month', 'Week', 'DOW'], axis=1)
        start_date = pd.to_datetime(self.predicted_date).date()
        end_date = start_date + timedelta(days=6)

        train = df.loc[df.index.date < start_date]
        # remove outliers from training set
        train = ForecastRunner.remove_outliers(train, fill)
        test = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        return train, test

    def grid_search(self, xtr, ytr):
        gbm = xgb.XGBRegressor()
        reg_cv = GridSearchCV(gbm,
                              {"colsample_bytree": self.colsample_bytree, "min_child_weight": self.min_child_weight,
                               'max_depth': self.max_depth, 'n_estimators': self.n_estimators}, verbose=1)
        reg_cv.fit(xtr, ytr)
        return reg_cv

    @staticmethod
    def plot_result(y_true, y_pred):
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.savefig('plot.png')

    def fit(self):
        """
        Gets data and preprocess by prepare_data() function
        Trains with the selected parameters from grid search and saves the model
        """
        data = self.get_input()
        df_train, df_test = self.prepare_data(data)
        xtr, ytr = df_train.drop(['Value'], axis=1), df_train['Value'].values

        xgbtrain = xgb.DMatrix(xtr, ytr)
        reg_cv = self.grid_search(xtr, ytr)
        param = reg_cv.best_params_
        bst = xgb.train(dtrain=xgbtrain, params=param)

        # save model to file
        mlflow.sklearn.save_model(bst, "model")
        return df_test

    def predict(self, df_test):
        """
         Makes prediction for the next 7 days electricity consumption.
        """
        # load model from file
        loaded_model = mlflow.sklearn.load_model("model")
        # make predictions for test data
        xts, yts = df_test.drop(['Value'], axis=1), df_test['Value'].values
        p = loaded_model.predict(xgb.DMatrix(xts))
        prediction = pd.DataFrame({'Prediction': p})

        mape, rmse, mae, r2 = ForecastRunner.evaluation_metrics(yts, p)
        print('MAPE: {}'.format(mape))
        print('RMSE: {}'.format(rmse))
        print('R2: {}'.format(r2))
        print('MAE: {}'.format(mae))
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        ForecastRunner.plot_result(yts, p)
        self.save_output(df_test, prediction)