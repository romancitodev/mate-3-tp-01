import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.analysis.classification import logistic_regression_three_classes


def analyze(df: pd.DataFrame):
    general(df)
    null_dup_revision(df)
    analyze_target(df)
    analyze_correlations(df)
    model, scaler, results = logistic_regression_three_classes(df)


def general(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("EXPLORACIÓN DEL DATASET")
    print("=" * 80)

    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head())

    print("\nEstadísticas:")
    print(round(df.describe().T, 2))


def null_dup_revision(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("CALIDAD DE DATOS")
    print("=" * 80)

    total = df.isnull().sum().sum()
    duplicados = df.duplicated().sum()

    print(f"\nValores faltantes: {total}")
    print(f"Filas duplicadas: {duplicados}")


def analyze_target(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("VARIABLE OBJETIVO: quality")
    print("=" * 80)

    print(f"\nRango: {df['quality'].min()} - {df['quality'].max()}")
    print(f"Media: {df['quality'].mean():.2f}")
    print("\nDistribución:")
    print(df["quality"].value_counts().sort_index())

    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    sns.countplot(data=df, x="quality", ax=axes[0, 0], color="#9F6BC2")
    axes[0, 0].set_title("Distribución de calidad")

    sns.boxplot(data=df, x="quality", ax=axes[0, 1], color="#9F6BC2")
    axes[0, 1].set_title("BoxPlot de calidad")

    sns.histplot(
        data=df,
        x="alcohol",
        bins=30,
        color="#9F6BC2",
        edgecolor="black",
        alpha=0.7,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Distribución del contenido de alcohol")

    quality_groups = df.groupby("quality")["alcohol"].mean()
    sns.lineplot(
        x=quality_groups.index,
        y=quality_groups.values,
        marker="o",
        linewidth=2,
        markersize=8,
        ax=axes[1, 1],
        color="#9F6BC2",
    )
    axes[1, 1].set_title("Alcohol Promedio por Calidad")

    plt.tight_layout()
    plt.show()


def analyze_correlations(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("CORRELACIONES")
    print("=" * 80)

    correlacion = df.corr()
    print("\nCorrelaciones con quality:")
    print(correlacion["quality"].sort_values(ascending=False))

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
    )
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.show()
