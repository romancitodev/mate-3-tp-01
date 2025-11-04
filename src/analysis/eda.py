import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.analysis.logistic_regression import from_datafame
from src.analysis.classification import logistic_regression_three_classes


def analyze(df: pd.DataFrame):
    general(df)
    null_dup_revision(df)
    analyze_target(df)
    analyze_correlations(df)

    # logistic_regression_three_classes(df)
    from_datafame(df)


def general(df: pd.DataFrame):
    def columnas(df: pd.DataFrame):
        cols = df.columns.tolist()
        print("\nColumnas:")
        print(f"╔{'═' * 32}╦{'═' * 32}╦{'═' * 32}╗")
        for i in range(0, len(cols) - 2, 3):
            print(f"║ {cols[i]:<31}║ {cols[i + 1]:<31}║ {cols[i + 2]:<31}║")
        print(f"╚{'═' * 32}╩{'═' * 32}╩{'═' * 32}╝\n")

    print("--- VISUALIZACION GENERAL ---".center(100))
    print("═" * 100)

    print("--- PRIMEROS REGISTROS ---")
    print(df.head())
    print("═" * 100)

    print(
        f"--- DIMENSIONES ---\n• {df.shape[1]:<8} Columnas \n• {df.shape[0]:<8} Filas"
    )
    columnas(df)
    print("═" * 100)

    print("--- INFORMACION GENERAL ---")
    print(f"{df.info()}")
    print("═" * 100)

    print("--- DESCRIPCION GENERAL ---")
    print(f"{round(df.describe().T, 2)}")
    print("═" * 100)


def null_dup_revision(df: pd.DataFrame):
    print("--- ANALISIS DE NULOS Y DUPLICADOS ---".center(100))
    print("═" * 100)

    print("--- VALORES FALTANTES ---")
    print(df.isnull().sum())
    total = df.isnull().sum().sum()
    print(f"\nTotal :{total}")
    if total:
        print("⚠ Existen valores faltantes que deben tratarse")
    else:
        print("✓ No hay valores faltantes")
    print("═" * 100)

    print("--- ANALISIS DUPLICADOS ---")
    duplicados = df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados}")
    print("═" * 100)


def analyze_target(df: pd.DataFrame):
    def graficar(df: pd.DataFrame):
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        # Distribución de calidad
        sns.countplot(data=df, x="quality", ax=axes[0, 0], color="#9F6BC2").set_title(
            "Distibucion de calidad"
        )

        # Boxplot de calidad
        sns.boxplot(data=df, x="quality", ax=axes[0, 1], color="#9F6BC2").set_title(
            "BoxPlot de calidad"
        )

        # Histograma de alcohol
        sns.histplot(
            data=df,
            x="alcohol",
            bins=30,
            color="#9F6BC2",
            edgecolor="black",
            alpha=0.7,
            ax=axes[1, 0],
        ).set_title("Distribucion del contenido de alcohol")

        # Relación alcohol vs calidad
        quality_groups = df.groupby("quality")["alcohol"].mean()
        sns.lineplot(
            x=quality_groups.index,
            y=quality_groups.values,
            marker="o",
            linewidth=2,
            markersize=8,
            ax=axes[1, 1],
            color="#9F6BC2",
        ).set_title("Alcohol Promedio por Calidad")

        plt.tight_layout()
        plt.show()

    print("--- ANALISIS DE VARIABLE OBJETIVO (quality) ---".center(100))
    print("═" * 100)

    print("--- DISTRIBUCION DE CALIDAD ---")
    print(f"Valores de calidad: {df['quality'].unique()}")
    print(df["quality"].value_counts())
    print(f"\nCalidad minima: {df['quality'].min()}")
    print(f"Calidad maxima: {df['quality'].max()}")
    print(f"Calidad media: {df['quality'].mean():.2f}")
    print(f"Calidad mediana: {df['quality'].median()}")
    graficar(df)

    print("═" * 100)


def analyze_correlations(df: pd.DataFrame):
    def graficar(df: pd.DataFrame):
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            correlacion,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
        )
        plt.title("Matriz de Correlación - Variables Fisicoquímicas del Vino")

        plt.show()

    print("--- CORRELACION ---".center(100))
    print("═" * 100)

    correlacion = df.corr()
    print("\nCorrelaciones con la variable quality:")
    print(correlacion["quality"])
    graficar(df)
    print("═" * 100)


def conclusions(df: pd.DataFrame):
    print("--- OBSERVACIONES ---".center(100))
    print("═" * 100)
