import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def train_and_evaluate_models(df: pd.DataFrame):
    """
    Entrena y evalúa múltiples modelos no lineales con datos escalados.
    """
    print("=" * 100)
    print("--- PREPARACIÓN DE DATOS Y ESCALADO ---".center(100))
    print("=" * 100)

    # Separar features (X) y target (y)
    X = df.drop(columns=["quality"]).values
    y = df["quality"].values

    # Mostrar información sobre los datos sin escalar
    print("\n--- RANGOS DE DATOS SIN ESCALAR ---")
    feature_names = df.drop(columns=["quality"]).columns.tolist()
    stats_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Min": X.min(axis=0),
            "Max": X.max(axis=0),
            "Mean": X.mean(axis=0),
            "Std": X.std(axis=0),
        }
    )
    print(stats_df.round(4))
    print("\n⚠ Las variables tienen rangos muy diferentes, necesitamos escalarlas")

    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n--- DIVISIÓN DE DATOS ---")
    print(f"• Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"• Conjunto de prueba: {X_test.shape[0]} muestras")

    # Aplicar StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- ESCALADO CON STANDARDSCALER ---")
    print("✓ StandardScaler aplicado (media=0, desviación estándar=1)")
    print("\nEstadísticas después del escalado (conjunto de entrenamiento):")
    scaled_stats = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean": X_train_scaled.mean(axis=0),
            "Std": X_train_scaled.std(axis=0),
            "Min": X_train_scaled.min(axis=0),
            "Max": X_train_scaled.max(axis=0),
        }
    )
    print(scaled_stats.round(4))

    print("\n" + "=" * 100)
    print("--- ENTRENAMIENTO DE MODELOS NO LINEALES ---".center(100))
    print("=" * 100)

    # Definir modelos no lineales
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1
        ),
        "Support Vector Machine": SVR(kernel="rbf", C=100, gamma="scale"),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
    }

    # Almacenar resultados
    results = []

    # Entrenar y evaluar cada modelo
    for name, model in models.items():
        print(f"\n--- Entrenando {name} ---")

        # Entrenar el modelo
        model.fit(X_train_scaled, y_train)

        # Predicciones
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Métricas de entrenamiento
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Métricas de prueba
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Validación cruzada
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")

        results.append(
            {
                "Modelo": name,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Train MAE": train_mae,
                "Test MAE": test_mae,
                "Train R²": train_r2,
                "Test R²": test_r2,
                "CV R² (mean)": cv_scores.mean(),
                "CV R² (std)": cv_scores.std(),
            }
        )

        print(f"✓ Entrenamiento completado")
        print(f"  • Train RMSE: {train_rmse:.4f}")
        print(f"  • Test RMSE:  {test_rmse:.4f}")
        print(f"  • Test R²:    {test_r2:.4f}")

    # Mostrar tabla comparativa
    print("\n" + "=" * 100)
    print("--- COMPARACIÓN DE MODELOS ---".center(100))
    print("=" * 100)

    results_df = pd.DataFrame(results)
    print("\n", results_df.round(4).to_string(index=False))

    # Encontrar el mejor modelo
    best_model_idx = results_df["Test R²"].idxmax()
    best_model_name = results_df.loc[best_model_idx, "Modelo"]
    best_r2 = results_df.loc[best_model_idx, "Test R²"]

    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   R² en conjunto de prueba: {best_r2:.4f}")

    # Visualizaciones
    visualize_results(results_df, models, X_test_scaled, y_test, feature_names)

    # Analizar importancia de features (para Random Forest y Gradient Boosting)
    analyze_feature_importance(models, feature_names)

    return models, scaler, results_df


def visualize_results(results_df, models, X_test_scaled, y_test, feature_names):
    """
    Visualiza los resultados de los modelos.
    """
    print("\n" + "=" * 100)
    print("--- VISUALIZACIONES ---".center(100))
    print("=" * 100)

    # Comparación de métricas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # RMSE Comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax1.bar(
        x_pos - width / 2,
        results_df["Train RMSE"],
        width,
        label="Train",
        color="#9F6BC2",
        alpha=0.8,
    )
    ax1.bar(
        x_pos + width / 2,
        results_df["Test RMSE"],
        width,
        label="Test",
        color="#6B9FC2",
        alpha=0.8,
    )
    ax1.set_xlabel("Modelo")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Comparación RMSE - Train vs Test")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df["Modelo"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # R² Comparison
    ax2 = axes[0, 1]
    ax2.bar(
        x_pos - width / 2,
        results_df["Train R²"],
        width,
        label="Train",
        color="#9F6BC2",
        alpha=0.8,
    )
    ax2.bar(
        x_pos + width / 2,
        results_df["Test R²"],
        width,
        label="Test",
        color="#6B9FC2",
        alpha=0.8,
    )
    ax2.set_xlabel("Modelo")
    ax2.set_ylabel("R² Score")
    ax2.set_title("Comparación R² - Train vs Test")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df["Modelo"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # MAE Comparison
    ax3 = axes[1, 0]
    ax3.bar(
        x_pos - width / 2,
        results_df["Train MAE"],
        width,
        label="Train",
        color="#9F6BC2",
        alpha=0.8,
    )
    ax3.bar(
        x_pos + width / 2,
        results_df["Test MAE"],
        width,
        label="Test",
        color="#6B9FC2",
        alpha=0.8,
    )
    ax3.set_xlabel("Modelo")
    ax3.set_ylabel("MAE")
    ax3.set_title("Comparación MAE - Train vs Test")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df["Modelo"], rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Cross-validation R² scores
    ax4 = axes[1, 1]
    ax4.bar(x_pos, results_df["CV R² (mean)"], color="#C26B9F", alpha=0.8)
    ax4.errorbar(
        x_pos,
        results_df["CV R² (mean)"],
        yerr=results_df["CV R² (std)"],
        fmt="none",
        color="black",
        capsize=5,
    )
    ax4.set_xlabel("Modelo")
    ax4.set_ylabel("R² Score")
    ax4.set_title("Cross-Validation R² (5-Fold)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results_df["Modelo"], rotation=45, ha="right")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Predicciones vs Valores Reales para el mejor modelo
    best_model_name = results_df.loc[results_df["Test R²"].idxmax(), "Modelo"]
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: Predicciones vs Valores Reales
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#9F6BC2", edgecolors="black")
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Predicción perfecta",
    )
    ax1.set_xlabel("Valores Reales")
    ax1.set_ylabel("Valores Predichos")
    ax1.set_title(f"Predicciones vs Reales - {best_model_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Distribución de errores
    ax2 = axes[1]
    errors = y_test - y_pred
    ax2.hist(errors, bins=30, color="#9F6BC2", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Error = 0")
    ax2.set_xlabel("Error de Predicción")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title(f"Distribución de Errores - {best_model_name}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_feature_importance(models, feature_names):
    """
    Analiza la importancia de las features para modelos basados en árboles.
    """
    print("\n" + "=" * 100)
    print("--- IMPORTANCIA DE FEATURES ---".center(100))
    print("=" * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Random Forest
    if "Random Forest" in models:
        rf_model = models["Random Forest"]
        importances_rf = rf_model.feature_importances_
        indices_rf = np.argsort(importances_rf)[::-1]

        ax1 = axes[0]
        ax1.barh(
            range(len(importances_rf)),
            importances_rf[indices_rf],
            color="#9F6BC2",
            alpha=0.8,
        )
        ax1.set_yticks(range(len(importances_rf)))
        ax1.set_yticklabels([feature_names[i] for i in indices_rf])
        ax1.set_xlabel("Importancia")
        ax1.set_title("Importancia de Features - Random Forest")
        ax1.grid(axis="x", alpha=0.3)

        print("\n--- Random Forest - Top 5 Features ---")
        for i in range(min(5, len(feature_names))):
            idx = indices_rf[i]
            print(f"{i + 1}. {feature_names[idx]}: {importances_rf[idx]:.4f}")

    # Gradient Boosting
    if "Gradient Boosting" in models:
        gb_model = models["Gradient Boosting"]
        importances_gb = gb_model.feature_importances_
        indices_gb = np.argsort(importances_gb)[::-1]

        ax2 = axes[1]
        ax2.barh(
            range(len(importances_gb)),
            importances_gb[indices_gb],
            color="#6B9FC2",
            alpha=0.8,
        )
        ax2.set_yticks(range(len(importances_gb)))
        ax2.set_yticklabels([feature_names[i] for i in indices_gb])
        ax2.set_xlabel("Importancia")
        ax2.set_title("Importancia de Features - Gradient Boosting")
        ax2.grid(axis="x", alpha=0.3)

        print("\n--- Gradient Boosting - Top 5 Features ---")
        for i in range(min(5, len(feature_names))):
            idx = indices_gb[i]
            print(f"{i + 1}. {feature_names[idx]}: {importances_gb[idx]:.4f}")

    plt.tight_layout()
    plt.show()
    print("=" * 100)
