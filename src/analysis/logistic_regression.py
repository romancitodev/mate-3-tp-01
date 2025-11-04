import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def from_datafame(df: pandas.DataFrame):
    # Separar características (X) y variable objetivo (y)
    X = df.drop(columns=["quality"]).values  # Todas las columnas excepto 'quality'
    y = df["quality"].values  # Solo la columna 'quality'

    # Dividir datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crear el modelo de regresión logística
    model = LogisticRegression(solver="liblinear", random_state=42)

    # Función para categorizar calidad del vino:
    # 0 = Malo (≤5), 1 = Bueno (6), 2 = Excelente (≥7)
    def categorize_quality(x):
        """Convierte calidad numérica a categorías: Malo(0), Bueno(1), Excelente(2)"""
        if x <= 4:
            return 0  # Malo
        elif x == 5:
            return 1  # Bueno
        else:
            return 2  # Excelente

    y_train = np.array(
        [categorize_quality(q) for q in y_train]
    )  # Categorizar datos de entrenamiento
    y_test = np.array(
        [categorize_quality(q) for q in y_test]
    )  # Categorizar datos de prueba

    # Normalizar las características para mejorar el rendimiento del modelo
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train
    )  # Ajustar scaler y transformar entrenamiento
    X_test = scaler.transform(X_test)  # Solo transformar prueba (usar mismo scaler)

    # Entrenar el modelo con datos normalizados y categorizados
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Crear y mostrar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)

    # Mostrar reporte de clasificación con métricas detalladas
    print_classification_report(y_test, y_pred)


def print_confusion_matrix(cm: np.ndarray):
    """
    Visualiza la matriz de confusión para clasificación de 3 clases
    """
    _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)

    # Configurar etiquetas para 3 clases: Malo(0), Bueno(1), Excelente(2)
    labels = ["Malo (≤4)", "Bueno (5)", "Excelente (≥6)"]
    ax.xaxis.set(ticks=(0, 1, 2), ticklabels=[f"Pred {label}" for label in labels])
    ax.yaxis.set(ticks=(0, 1, 2), ticklabels=[f"Real {label}" for label in labels])
    ax.set_ylim(2.5, -0.5)  # Ajustar límites para 3 clases

    # Mostrar valores en cada celda de la matriz
    for i in range(cm.shape[0]):  # Usar shape real de la matriz
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )  # Contraste automático

    plt.title("Matriz de Confusión - Clasificación de Calidad del Vino")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores Reales")
    plt.show()


def print_classification_report(y_test, y_pred):
    """
    Imprime el reporte de clasificación para el modelo
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1-Score", f1),
    ]

    # Crear un subplot para cada métrica
    for idx, (ax, (metric_name, metric_value)) in enumerate(zip(axes.flat, metrics)):
        ax.barh(
            [metric_name],
            [metric_value],
            color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][idx],
        )
        ax.set_xlim(0, 1)
        ax.set_title(
            f"{metric_name}: {metric_value:.4f}", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Métricas de Clasificación", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
