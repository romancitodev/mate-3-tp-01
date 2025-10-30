import pandas as pd
import matplotlib.pyplot as plt


def analysis_eda(df: pd.DataFrame):
    general(df)
    null_revision(df)
    #visualize_nulls(df)
    df = clean_data(df)


def general(df: pd.DataFrame):
    # ==========================================
    #             REVISION GENERAL
    # ==========================================
    print("--- REVISION GENERAL ---".center(100))
    print("═" * 100)
    print(f"--- PRIMEROS REGISTROS ---\n{df.head()}")
    print("═" * 100)

    print(
        f"--- DIMENSIONES ---\n• {df.shape[1]:<8} Columnas \n• {df.shape[0]:<8} Filas"
    )
    show_columns(df)
    print("═" * 100)

    print("--- INFORMACION GENERAL ---")
    print(f"{df.info()}")
    print("═" * 100)

    print("""
    --- OBSERVACIONES ANALISIS GENERAL ---

    - Varias columnas con valores nulos
    - Varias columnas con Informacion poco util para el analisis (movies, screenshots, URLs, etc.)
    """)
    print("═" * 100)


def show_columns(df: pd.DataFrame):
    columns = df.columns.tolist()
    print("\nColumnas:")
    print(f"{'═' * 94}╗")
    for i in range(0, len(columns) - 2, 3):
        print(f"{columns[i]:<30}║ {columns[i + 1]:<30}║ {columns[i + 2]:<30}║")
    print(f"{'═' * 94}╝\n")


def null_revision(df: pd.DataFrame):
    # ==========================================
    #    REVISION DATOS FALTANTES Y DUPLICADOS
    # ==========================================
    print("--- REVISION DE DATOS FALTANTES Y DUPLICADOS ---".center(100))
    print("═" * 100)

    nulos_por_col = df.isnull().sum()
    nulos_por_col = nulos_por_col[nulos_por_col > 0]

    resumen_nulos = pd.DataFrame(
        {
            "cant_nulos": nulos_por_col,
            "porcentaje_nulos": ((nulos_por_col / len(df)) * 100).round(2),
        }
    )

    total_nulos = df.isnull().sum().sum()
    porcentaje_total = (total_nulos / df.size) * 100

    print("--- VALORES NULOS ---\n")
    print("Por columnas:")
    print(resumen_nulos)
    print("\nTotales:")
    print(f"Nulos totales: {total_nulos} ({porcentaje_total:.2f}%)")
    print("═" * 100)

    print("--- VERIFICACION DE CEROS SOSPECHOSOS ---")

    columnas_con_listas = [
        "full_audio_languages",
        "supported_languages",
        "screenshots",
        "movies",
        "tags",
    ]

    print("\nColumnas con listas:")
    for col in columnas_con_listas:
        vacias = (df[col] == "[]").sum()
        porcentaje = (vacias / len(df)) * 100
        print(f" • {col}: {vacias} listas vacías ({porcentaje:.2f}%)")

    columnas_playtime = [
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
    ]

    print("\nPlayTime = 0:")
    for col in columnas_playtime:
        playtime_cero = (df[col] == 0).sum()
        porcentaje = (playtime_cero / len(df)) * 100
        print(f" • {col}: {playtime_cero} ({porcentaje:.2f}%)")
    print("═" * 100)

    print("---  VALORES DUPLICADOS ---")
    print(f"{df.duplicated()}\n")
    print(f"Totales: {df.duplicated().sum()}")
    print("═" * 100)
    print("""
    --- OBSERVACIONES ANALISIS DE NULOS Y DUPLICADOS ---

    - Varias columnas con un alto porcentaje de valores nulos.
    - Sin registros duplicados

    Tratamiento de datos.
    Al ser valores prescindibles para el analisis, descripciones y URLs en su mayoria.
    Se eliminaran la columnas conflictivas, junto a otra innecesarias.
    """)
    print("═" * 100)


def visualize_nulls(df: pd.DataFrame):
    """Crear gráfico de barras con porcentaje de nulos por columna"""

    nulos_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    nulos_pct = nulos_pct[nulos_pct > 0]  # Solo columnas con nulos

    if len(nulos_pct) > 0:
        plt.figure(figsize=(12, 6))
        nulos_pct.plot(kind="barh", color="coral")
        plt.xlabel("Porcentaje de Valores Nulos (%)")
        plt.ylabel("Columnas")
        plt.title("Porcentaje de Valores Nulos por Columna")
        plt.axvline(x=50, color="red", linestyle="--", label="50% threshold")
        plt.legend()
        plt.tight_layout()
        plt.show()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("--- LIMPIAR DATOS ---".center(100))
    print("═" * 100)

    columnas_eliminar = [
        "appid",
        "name",
        "detailed_description",
        "about_the_game",
        "short_description",
        "reviews",
        "header_image",
        "support_url",
        "support_email",
        "metacritic_url",
        "notes",
        "full_audio_languages",
        "screenshots",
        "movies",
        "socore_rank", # 99.98% de Valores Nulos
        "average_playtime_forever", # 91.06% de Valores Nulos
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
        "peak_ccu",
    ]
    print(f"Eliminando {len(columnas_eliminar)} columnas innecesarias...")
    df_clean = df.drop(columns=columnas_eliminar, errors="ignore")
    print(f"Columnas restantes: {df_clean.shape[1]}")
    
    idioma_comun = df_clean["supported_languages"].value_counts().idxmax()
    print(f"Valor mas comun de Idiomas: {type(idioma_comun)}")
    
    df_clean["supported_languages"] = df_clean["supported_languages"].apply(lambda x: idioma_comun if x == "[]" else x)
    
    print(f"Valor mas comun de Tags: {df_clean[df_clean["tags"].apply(lambda x: x != "[]")]["tags"].value_counts().idxmax()}")
    
    
    columnas_con_listas = [
        "supported_languages",
        "tags",
    ]

    print("\nColumnas con listas:")
    for col in columnas_con_listas:
        vacias = (df_clean[col] == "[]").sum()
        porcentaje = (vacias / len(df)) * 100
        print(f" • {col}: {vacias} listas vacías ({porcentaje:.2f}%)")
    
    

    return df_clean
