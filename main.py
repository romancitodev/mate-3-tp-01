from pandas import read_csv
from src.analysis.eda import analyze
from src.screen import clear_screen
from src.config import configure
from src.model.logistic_regression_ import logistic_regression

from src.model.classification import logistic_regression_three_classes


def main():
    clear_screen()
    configure()

    df = read_csv("data/wine.csv", encoding="UTF-8")

    print("═" * 100)
    print("ANÁLISIS DE CALIDAD DE VINOS".center(100))
    print("═" * 100)

    df, x_train, x_test, y_train, y_test, class_names = analyze(df)
    
    logistic_regression(df, x_train, y_train, x_test, y_test, class_names)
    #logistic_regression_three_classes(df)


if __name__ == "__main__":
    main()
