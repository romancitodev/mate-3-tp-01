import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analyze(df: pd.DataFrame):
    
    print("═" * 100)
    print("ANALISIS EXPLORATORIO".center(100))
    print("═" * 100)
    
    general(df)
    null_dup_revision(df)
    analyze_target(df)
    analyze_correlations(df)
    conclusions(df)
    
    return prepare_data(df)



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

    print(f"--- DIMENSIONES ---")
    print(f"• {df.shape[1]:<8} Columnas \n• {df.shape[0]:<8} Filas")
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
        print("Existen valores faltantes que deben tratarse")
    else:
        print("No hay valores faltantes")
    print("═" * 100)

    print("--- ANALISIS DUPLICADOS ---")
    duplicados = df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados} ({(duplicados*100/len(df)):.2f}%)")
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
        plt.figure(figsize=(10, 7))
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
    print("Correlaciones con la variable quality:")
    print(correlacion["quality"])
    graficar(df)
    print("═" * 100)


def conclusions(df: pd.DataFrame):
    print("--- OBSERVACIONES ---".center(100))
    print("═" * 100)
    
    duplicados = df.duplicated().sum()
    print("--- CARACTERISTICAS GENERALES ---")
    print(f"""
          • Tamaño: {len(df)} Filas x {len(df.columns)} Columnas
          • No hay valores Nulos.
          • Filas duplicadas: {duplicados} ({(duplicados*100/len(df)):.2f}%)")
          • Todas variables numericas.
          """)
    
    print("--- SOBRE CALIDAD ---")
    print(f"""
          • Rango: {df['quality'].min()} a {df['quality'].max()}
          • Distribucion concentrada en valores medios (5-6).
          • Todas variables enteros numericos.
          """)
    
    correlacion = df.corr()["quality"]
    print("--- CORRELACIONES PRINCIPALES CON CALIDAD ---")
    print(f"""
          • Alcohol: {correlacion["alcohol"]:.2f}. (Mas alta)
          • Acidez Volatil: {correlacion["volatile acidity"]:.2f}. (Mas Baja)
          • Sulfatos: {correlacion["sulphates"]:.2f}. (Valor Medio)
          """)
    
    print("═" * 100)
    
def prepare_data(df: pd.DataFrame):
    print("--- PREPARACIÓN DE DATOS ---".center(100))
    print("═" * 100)
    
    df = df.drop_duplicates()
    print(f"Se borraron registros duplicados")
    print("═" * 100)
    
    X = df.drop(columns=["quality"]).values
    y = df["quality"].values
    feature_names = df.drop(columns=["quality"]).columns.tolist()

    def categorize_quality(q):
        """
        Categorización balanceada de calidad:
        - Bajo (0): ≤5
        - Medio (1): 6
        - Alto (2): ≥7
        """
        if q <= 5:
            return 0
        elif q == 6:
            return 1
        else:
            return 2

    y_categorical = np.array([categorize_quality(q) for q in y])
    class_names = ["Bajo (≤5)", "Medio (6)", "Alto (≥7)"]
    
    print("Calidad categorizada")
    print("\nDistribución de categorías:")
    unique, counts = np.unique(y_categorical, return_counts=True)
    for cls_idx, (cls_name, count) in enumerate(zip(class_names, counts)):
        percentage = (count / len(y_categorical)) * 100
        print(f"  {cls_name}: {count} muestras ({percentage:.2f}%)")
    print("═" * 100)
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.2, 
        random_state=42,
        stratify=y_categorical
    )
    print("Se crearon los datos de entrenamiento y prueba:")
    print(f"\nDatos de entrenamiento: {len(x_train)} muestras")
    print(f"Datos de prueba: {len(x_test)} muestras")
    print("═" * 100)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("Normalizacion aplicada (StandardScaler)")
    print("═" * 100)
    
    return df, x_train_scaled, x_test_scaled, y_train, y_test, class_names