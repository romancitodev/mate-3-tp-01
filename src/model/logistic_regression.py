import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def logistic_regression(
    df: pd.DataFrame, x_train, y_train, x_test, y_test, class_names
):
    print("\n\n" + "═" * 100)
    print("REGRESIÓN LOGÍSTICA - CLASIFICACIÓN EN 3 CATEGORÍAS".center(100))
    print("═" * 100)

    model = prepare_model(x_train, y_train)

    # Validación cruzada para verificar que el modelo generaliza bien y no solo memoriza
    cv_score = cross_val_score(model, x_train, y_train)

    cm, y_pred = prediction(model, x_test, y_test, class_names)

    graficar(y_pred, cm, cv_score, class_names)

    return model, {"cv_score": cv_score, "cm": cm, "y_pred": y_pred}


def prepare_model(x_train, y_train):
    print("--- MODELO ---".center(100))
    print("═" * 100)

    print("""
    Debido al desbalanceo del DataSet (Mayoria en 'Bajo' o 'Medio')
    Se aplicara 'class_weight' para compensar.
    Tras analizar varias opciones de configuracion, se eligio el modo balanceado.
    """)

    # Para este problema de clasificación multi-clase usamos Regresión Logística
    # porque necesitamos predecir categorías discretas (Bajo/Medio/Alto), no valores continuos.
    # El modelo aplicará una función sigmoide para calcular probabilidades de pertenencia
    # a cada clase, permitiendo asignar cada vino a la categoría más probable.
    #
    # Configuramos 'multinomial' ya que tenemos 3 clases, y usamos 'lbfgs' como solver
    # porque es eficiente para datasets de este tamaño y maneja bien el problema multi-clase.

    model = LogisticRegression(
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    print("Entranando modelo ...")
    model.fit(x_train, y_train)

    print("═" * 100)

    return model


def cross_validation(model, x_train, y_train):
    print("--- VALIDACIÓN CRUZADA (5-FOLD) ---".center(100))
    print("═" * 100)

    # La validación cruzada divide los datos en 5 partes y entrena 5 veces, cada vez usando
    # una parte diferente como test. Esto nos da una estimación más robusta del rendimiento
    # real del modelo y nos permite detectar si está sobreajustando a los datos de entrenamiento.
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")

    print(f"\nAccuracy por fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Accuracy media: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")
    print("═" * 100)

    return cv_scores


def prediction(model, x_test, y_test, class_names):
    print("--- PREDICCIONES ---".center(100))
    print("═" * 100)

    y_pred = model.predict(x_test)

    # Calculamos múltiples métricas para evaluar el rendimiento desde distintos ángulos:
    # accuracy nos da la visión general, pero precision/recall/f1 son cruciales para
    # detectar si el modelo está sesgado hacia alguna clase en particular (especialmente
    # importante dado el desbalanceo del dataset).
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # pyright: ignore[reportArgumentType]
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("--- METRICAS ---")
    print(f"• Accuracy:  {accuracy:.4f}")
    print(f"• Precision: {precision:.4f}")
    print(f"• Recall:    {recall:.4f}")
    print(f"• F1-Score:  {f1:.4f}")
    print("═" * 100)

    # El rendimiento por clase nos revela patrones importantes: las categorías extremas
    # (Bajo/Alto) suelen tener mejor accuracy (~72%) que la categoría Media (~39%).
    # Esto se debe a que hay menos muestras de calidad media y además las características
    # fisicoquímicas de vinos "medianos" son más ambiguas y difíciles de separar.
    print("--- RENDIMIENTO POR CLASE ---")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"  {name}: {correct}/{total} correctos ({acc:.2f}%)")
    print("═" * 100)

    print("--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
    print(
        classification_report(y_test, y_pred, target_names=class_names, zero_division=0)  # pyright: ignore[reportArgumentType]
    )

    return cm, y_pred


def graficar(y_pred, cm, cv_scores, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]  # Rojo: Bajo, Azul: Medio, Verde: Alto

    # Matriz de confusión - muestra dónde el modelo se equivoca más frecuentemente
    ax1 = axes[0, 0]
    sns.heatmap(
        cm,
        ax=ax1,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    ax1.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Clase Real", fontsize=11)
    ax1.set_xlabel("Clase Predicha", fontsize=11)

    # Distribución de predicciones
    unique, counts = np.unique(y_pred, return_counts=True)

    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(unique)), counts, color=colors)

    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel("Cantidad de Predicciones", fontsize=11)
    ax2.set_title("Distribución de Predicciones", fontsize=13, fontweight="bold")
    # Agregar valores en las barras
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{int(bar.get_height())}",
            ha="center",
            fontweight="bold",
            color="white",
        )

    # Accuracy por clase
    class_accs = [
        (cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0)
        for i in range(len(class_names))
    ]

    ax3 = axes[1, 0]
    bars3 = ax3.barh(
        range(len(class_names)),
        class_accs,
        color=colors,
    )

    ax3.set_yticks(range(len(class_names)))
    ax3.set_yticklabels(class_names)
    ax3.set_xlabel("Accuracy", fontsize=11)
    ax3.set_title("Accuracy por Clase", fontsize=13, fontweight="bold")
    ax3.set_xlim([0, 1])
    # Agregar valores en las barras
    for bar, acc in zip(bars3, class_accs):
        ax3.text(
            bar.get_width() - 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2%}",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Cross-validation - si los 5 folds tienen accuracy similar, el modelo es estable
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(1, len(cv_scores) + 1), cv_scores, color="#9F6BC2")

    # La línea roja muestra el promedio: ~60% de accuracy es consistente entre folds
    ax4.axhline(
        cv_scores.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Media: {cv_scores.mean():.3f}",
    )

    ax4.set_xlabel("Fold", fontsize=11)
    ax4.set_ylabel("Accuracy", fontsize=11)
    ax4.set_title("Cross-Validation (5-Fold)", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1])
    # Agregar valores en las barras
    for bar, score in zip(bars4, cv_scores):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="white",
            fontsize=9,
        )

    plt.suptitle(
        "Análisis de Regresión Logística - Clasificación de Vinos",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.show()
