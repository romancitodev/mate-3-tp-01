from pandas import read_csv
from src.analysis import eda
from src.screen import clear_screen
from src.config import configure

UTF8 = "UTF-8"


def main():
    clear_screen()
    configure()

    df = read_csv("data/wine.csv", encoding=UTF8, sep=",")

    print("=" * 80)
    print("ANÁLISIS Y PREDICCIÓN DE CALIDAD DE VINOS")
    print("=" * 80)
    eda.analysis_eda(df)


if __name__ == "__main__":
    main()
