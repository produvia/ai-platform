from runner import ForecastRunner
import click


URL = 'https://data.smartdublin.ie/dataset/10130831-d2a5-4f4f-b56b-03d8e2cc56c8/resource/' \
    '6091c604-8c94-4b44-ac52-c1694e83d746/download/dccelectricitycivicsblocks34p20130221-1840.csv'


@click.command()
@click.option("--url", default=URL)
@click.option("--output_file", default='../output/output.csv')
@click.option("--predicted_date", default='2013-01-07')
@click.option("--min_child_weight", default=[0.8,1.2])
@click.option("--colsample_bytree", default=[0.9])
@click.option("--max_depth", default=[3, 4, 6])
@click.option("--n_estimators", default=[500, 1000])
@click.option("--eval_metric", default=['rmse'])
def main(url, output_file, predicted_date, min_child_weight, colsample_bytree, max_depth, n_estimators,
                 eval_metric):
    model = ForecastRunner(url, output_file, predicted_date, min_child_weight, colsample_bytree, max_depth, n_estimators,
                 eval_metric)
    df_test = model.fit()
    model.predict(df_test)


if __name__ == '__main__':
    main()
