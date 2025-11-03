import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.analysis.modeling import train_and_evaluate_models


def analyze(df: pd.DataFrame):
    general(df)
    null_dup_revision(df)
    analyze_target(df)
    analyze_correlations(df)

    # Entrenar y evaluar modelos con datos escalados
    print("\n")
    models, scaler, results = train_and_evaluate_models(df)

    conclusions(results)


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


def conclusions(results_df: pd.DataFrame):
    """
    Muestra conclusiones basadas en los resultados del modelado.
    """
    print("\n" + "=" * 100)
    print("--- CONCLUSIONES Y RECOMENDACIONES ---".center(100))
    print("=" * 100)

    best_model_idx = results_df["Test R²"].idxmax()
    best_model = results_df.loc[best_model_idx, "Modelo"]
    best_r2 = results_df.loc[best_model_idx, "Test R²"]
    best_rmse = results_df.loc[best_model_idx, "Test RMSE"]

    print("\n📊 RESUMEN DE RESULTADOS:")
    print(f"\n1. ESCALADO DE DATOS:")
    print("   ✓ Se aplicó StandardScaler a todas las features")
    print(
        "   ✓ Las variables tenían rangos muy diferentes (ej: density ~0.99 vs total sulfur dioxide ~34)"
    )
    print("   ✓ Después del escalado: media=0, desviación estándar=1")

    print(f"\n2. MEJOR MODELO:")
    print(f"   🏆 {best_model}")
    print(f"   • R² Score: {best_r2:.4f}")
    print(f"   • RMSE: {best_rmse:.4f}")

    print(f"\n3. COMPARACIÓN DE MODELOS:")
    print("   • Random Forest y Gradient Boosting suelen ser los más robustos")
    print("   • SVM puede ser sensible a la elección de hiperparámetros")
    print("   • Neural Networks pueden requerir más datos o ajuste fino")

    print(f"\n4. RECOMENDACIONES:")
    if best_r2 > 0.4:
        print("   ✓ El modelo tiene capacidad predictiva aceptable")
    else:
        print("   ⚠ El R² es bajo, considerar:")
        print("     - Feature engineering adicional")
        print("     - Optimización de hiperparámetros")
        print("     - Recolectar más datos")

    print("\n5. PRÓXIMOS PASOS:")
    print(
        "   • Optimizar hiperparámetros del mejor modelo (GridSearchCV/RandomizedSearchCV)"
    )
    print("   • Probar ensemble methods combinando múltiples modelos")
    print("   • Analizar casos donde el modelo falla más")
    print("   • Considerar tratamiento de outliers si existen")

    print("\n" + "=" * 100)
