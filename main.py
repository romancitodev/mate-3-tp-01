from pandas import read_csv

from src.analysis.eda import analyze
from src.config import configure
from src.model.logistic_regression import logistic_regression
from src.screen import clear_screen


def main():
    # Este proyecto implementa clasificación de calidad de vinos usando Regresión Logística.
    # Objetivo: Predecir si un vino es de calidad Baja (≤5), Media (6) o Alta (≥7)
    # basándonos en 11 características fisicoquímicas del dataset Wine Quality.
    # Elegimos regresión logística porque es un problema de clasificación multi-clase,
    # no de predicción de valores continuos (donde usaríamos regresión lineal).

    clear_screen()
    configure()

    df = read_csv("data/wine.csv", encoding="UTF-8")

    print("═" * 100)
    print("ANÁLISIS DE CALIDAD DE VINOS".center(100))
    print("═" * 100)

    # Primero realizamos EDA para entender correlaciones (alcohol es la más fuerte)
    df, x_train, x_test, y_train, y_test, class_names = analyze(df)

    # Luego entrenamos el modelo logístico y evaluamos con múltiples métricas
    model, data = logistic_regression(df, x_train, y_train, x_test, y_test, class_names)

    return 0


if __name__ == "__main__":
    main()
