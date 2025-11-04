from pandas import read_csv
from src.analysis.eda import analyze
from src.screen import clear_screen
from src.config import configure


def main():
    clear_screen()
    configure()

    df = read_csv("data/wine.csv", encoding="UTF-8")

    print("═" * 100)
    print("ANÁLISIS DE CALIDAD DE VINOS".center(100))
    print("═" * 100)

    analyze(df)


if __name__ == "__main__":
    main()
