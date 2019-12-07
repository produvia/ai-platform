from runner import ForecastRunner
import mlflow
import sys


if __name__ == "__main__":
    url = sys.argv[1]
    output_file = sys.argv[2]
    predicted_date = sys.argv[3]
    min_child_weight = eval(sys.argv[4])
    colsample_bytree = eval(sys.argv[5])
    max_depth = eval(sys.argv[6])
    n_estimators = eval(sys.argv[7])
    with mlflow.start_run():
        model = ForecastRunner(url, output_file, predicted_date, min_child_weight, colsample_bytree, max_depth,
                               n_estimators)
        df_test = model.fit()
        model.predict(df_test)
