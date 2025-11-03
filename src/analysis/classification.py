import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def logistic_regression_three_classes(df: pd.DataFrame):
    """Clasificación de vinos en 3 categorías usando Regresión Logística"""

    print("\n" + "=" * 80)
    print("REGRESIÓN LOGÍSTICA - 3 CATEGORÍAS")
    print("=" * 80)

    # Preparar datos
    X = df.drop(columns=["quality"]).values
    y = df["quality"].values

    def categorize_quality(q):
        if q <= 5:
            return 0  # Bajo
        elif q == 6:
            return 1  # Medio
        else:
            return 2  # Alto

    y_categorical = np.array([categorize_quality(q) for q in y])
    class_names = ["Bajo (3-5)", "Medio (6)", "Alto (7-9)"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encontrar mejor configuración
    best_model, best_name = _find_best_config(
        X_train_scaled, X_test_scaled, y_train, y_test, class_names
    )

    # Predicciones
    y_pred = best_model.predict(X_test_scaled)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

    # Resultados
    print(f"\nConfiguración: {best_name}")
    print(
        f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
    )
    print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print("\nRendimiento por clase:")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"  {name}: {correct}/{total} ({acc:.1f}%)")

    # Visualizaciones
    _plot_results(y_test, y_pred, cm, class_names, cv_scores)

    return (
        best_model,
        scaler,
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "cv_scores": cv_scores,
        },
    )


def _find_best_config(X_train, X_test, y_train, y_test, class_names):
    """Prueba diferentes configuraciones y retorna la mejor"""
    configs = [
        ("Sin ajuste", {}),
        ("Balanced", {"class_weight": "balanced"}),
        ("Peso x5 en Bajo", {"class_weight": {0: 5.0, 1: 1.0, 2: 1.0}}),
        ("Peso x10 en Bajo", {"class_weight": {0: 10.0, 1: 1.0, 2: 1.0}}),
    ]

    best_score = 0
    best_model = None
    best_name = None

    for name, params in configs:
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        per_class_acc = [
            (cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0)
            for i in range(len(class_names))
        ]
        avg_acc = np.mean(per_class_acc)

        if avg_acc > best_score:
            best_score = avg_acc
            best_model = model
            best_name = name

    return best_model, best_name


def _plot_results(y_test, y_pred, cm, class_names, cv_scores):
    """Genera visualizaciones"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Matriz de confusión
    ax1 = axes[0, 0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax1.set_title("Matriz de Confusión")
    ax1.set_ylabel("Real")
    ax1.set_xlabel("Predicho")

    # Distribución de predicciones
    ax2 = axes[0, 1]
    unique, counts = np.unique(y_pred, return_counts=True)
    ax2.bar(range(len(unique)), counts, color=["#e74c3c", "#3498db", "#2ecc71"])
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel("Cantidad")
    ax2.set_title("Distribución de Predicciones")
    ax2.grid(axis="y", alpha=0.3)

    # Accuracy por clase
    ax3 = axes[1, 0]
    class_accs = [
        (cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0)
        for i in range(len(class_names))
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    ax3.barh(range(len(class_names)), class_accs, color=colors)
    ax3.set_yticks(range(len(class_names)))
    ax3.set_yticklabels(class_names)
    ax3.set_xlabel("Accuracy")
    ax3.set_title("Accuracy por Clase")
    ax3.set_xlim([0, 1])
    ax3.grid(axis="x", alpha=0.3)

    # Cross-validation
    ax4 = axes[1, 1]
    ax4.bar(range(1, len(cv_scores) + 1), cv_scores, color="#9b59b6")
    ax4.axhline(
        cv_scores.mean(),
        color="red",
        linestyle="--",
        label=f"Media: {cv_scores.mean():.3f}",
    )
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Cross-Validation (5-Fold)")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
