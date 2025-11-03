from pandas import read_csv
from src.analysis import eda
from src.screen import clear_screen
from src.config import configure


def main():
    clear_screen()
    configure()

    df = read_csv("data/wine.csv", encoding="UTF-8")

    print("=" * 80)
    print("ANÁLISIS DE CALIDAD DE VINOS")
    print("=" * 80)

    eda.analyze(df)


if __name__ == "__main__":
    main()
