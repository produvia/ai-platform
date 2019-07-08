from runner import ForecastRunner
import sys


def main():
    url = sys.argv[1]
    output_file = sys.argv[2]
    predicted_date = sys.argv[3]
    model = ForecastRunner(url, output_file, predicted_date)
    df_test = model.fit()
    model.predict(df_test)


if __name__ == '__main__':
    main()

