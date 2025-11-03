# Modelado y Escalado de Datos - Wine Quality Prediction

## 📊 Descripción General

Este documento explica la estrategia de escalado y modelado implementada para predecir la calidad del vino basándose en sus características fisicoquímicas.

## 🔄 Escalado de Datos con StandardScaler

### ¿Por qué necesitamos escalar?

Los datos del dataset de vinos tienen rangos muy diferentes:

| Variable | Rango Aproximado | Escala |
|----------|-----------------|--------|
| `density` | 0.99 - 1.00 | Muy pequeña |
| `pH` | 2.5 - 4.0 | Pequeña |
| `chlorides` | 0.01 - 0.6 | Pequeña |
| `alcohol` | 8 - 15 | Media |
| `total sulfur dioxide` | 6 - 289 | Grande |
| `free sulfur dioxide` | 1 - 72 | Media-Grande |

**Problema**: Modelos como SVM, KNN y Neural Networks son sensibles a la escala de las variables. Sin escalado, las variables con rangos mayores dominarían el modelo.

### StandardScaler: ¿Qué hace?

StandardScaler transforma cada feature para tener:
- **Media (μ) = 0**
- **Desviación estándar (σ) = 1**

**Fórmula**:
```
z = (x - μ) / σ
```

### Implementación

```python
from sklearn.preprocessing import StandardScaler

# Crear el scaler
scaler = StandardScaler()

# Ajustar y transformar datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Solo transformar datos de prueba (usar parámetros del train)
X_test_scaled = scaler.transform(X_test)
```

**⚠️ IMPORTANTE**: 
- Solo aplicamos `fit_transform()` en el conjunto de **entrenamiento**
- En el conjunto de **prueba** solo usamos `transform()`
- Esto evita **data leakage** (filtración de información)

### ¿Qué variables escalamos?

**SÍ escalamos**: Todas las features (X)
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`

**NO escalamos**: La variable objetivo (y)
- `quality` - Se mantiene en su escala original (3-9)

## 🤖 Modelos No Lineales

### ¿Por qué modelos no lineales?

La relación entre las características químicas y la calidad del vino **no es lineal**. Un modelo lineal asume que:

```
quality = β₀ + β₁·alcohol + β₂·pH + ... + ε
```

Pero en realidad, las interacciones son más complejas:
- El efecto del alcohol puede depender del pH
- Múltiples variables interactúan entre sí
- Hay relaciones no lineales (cuadráticas, exponenciales, etc.)

### Modelos Implementados

#### 1. **Random Forest** 🌲
```python
RandomForestRegressor(n_estimators=100, max_depth=10)
```

**Ventajas**:
- ✅ Captura relaciones no lineales complejas
- ✅ Robusto a outliers
- ✅ Muestra importancia de features
- ✅ Reduce overfitting mediante averaging

**Cómo funciona**: Crea múltiples árboles de decisión y promedia sus predicciones.

#### 2. **Gradient Boosting** 📈
```python
GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
```

**Ventajas**:
- ✅ Muy alta precisión
- ✅ Aprende de errores previos
- ✅ Muestra importancia de features

**Cómo funciona**: Construye árboles secuencialmente, donde cada árbol corrige errores del anterior.

#### 3. **Support Vector Machine (SVM)** 🎯
```python
SVR(kernel='rbf', C=100, gamma='scale')
```

**Ventajas**:
- ✅ Efectivo en espacios de alta dimensión
- ✅ Kernel RBF captura relaciones no lineales

**Consideración**: Muy sensible al escalado (por eso es crucial StandardScaler).

#### 4. **K-Nearest Neighbors (KNN)** 👥
```python
KNeighborsRegressor(n_neighbors=10, weights='distance')
```

**Ventajas**:
- ✅ Simple e interpretable
- ✅ No asume forma de la relación

**Consideración**: También muy sensible al escalado.

#### 5. **Neural Network (MLP)** 🧠
```python
MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
```

**Ventajas**:
- ✅ Captura patrones muy complejos
- ✅ Flexible y poderoso

**Consideración**: Puede requerir más datos y tuning.

## 📏 Métricas de Evaluación

### RMSE (Root Mean Squared Error)
```
RMSE = √(Σ(y_real - y_pred)² / n)
```
- **Interpretación**: Error promedio en la escala de quality
- **Ejemplo**: RMSE = 0.6 → error promedio de ±0.6 puntos de calidad

### MAE (Mean Absolute Error)
```
MAE = Σ|y_real - y_pred| / n
```
- **Interpretación**: Error absoluto promedio
- **Más robusto** a outliers que RMSE

### R² Score (Coeficiente de Determinación)
```
R² = 1 - (SS_residual / SS_total)
```
- **Rango**: -∞ a 1
- **Interpretación**: 
  - R² = 1: Predicción perfecta
  - R² = 0.7: El modelo explica 70% de la varianza
  - R² = 0: Modelo tan bueno como predecir la media
  - R² < 0: Modelo peor que predecir la media

## 🔍 Interpretación de Resultados

### Ejemplo de salida esperada:

```
Modelo                    Test RMSE  Test MAE  Test R²
Random Forest            0.5234     0.3987    0.4521
Gradient Boosting        0.5187     0.3912    0.4589
Support Vector Machine   0.5567     0.4234    0.3876
K-Nearest Neighbors      0.5891     0.4567    0.3234
Neural Network           0.5423     0.4123    0.4201
```

**Interpretación**:
- **Gradient Boosting** tiene el mejor R² (0.4589)
- Puede predecir quality con error promedio de ~0.52 puntos
- El modelo explica ~46% de la varianza en quality

### Importancia de Features

Los modelos basados en árboles (RF, GB) muestran qué variables son más importantes:

```
Top 5 Features importantes:
1. alcohol: 0.2345
2. volatile acidity: 0.1876
3. sulphates: 0.1234
4. total sulfur dioxide: 0.0987
5. density: 0.0845
```

## 🎯 Validación Cruzada

Se implementa **5-Fold Cross-Validation**:

```python
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
```

**Beneficios**:
- Evalúa el modelo en diferentes subconjuntos
- Reduce varianza en la estimación del rendimiento
- Detecta overfitting

## 📊 Visualizaciones Generadas

1. **Comparación de RMSE** (Train vs Test)
2. **Comparación de R²** (Train vs Test)
3. **Comparación de MAE** (Train vs Test)
4. **Cross-Validation R²** con barras de error
5. **Predicciones vs Valores Reales** (scatter plot)
6. **Distribución de Errores** (histograma)
7. **Importancia de Features** (Random Forest y Gradient Boosting)

## 🚀 Próximos Pasos

### 1. Optimización de Hiperparámetros
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
```

### 2. Feature Engineering
- Crear interacciones entre variables
- Transformaciones polinomiales
- Binning de variables continuas

### 3. Ensemble Methods
- Combinar predicciones de múltiples modelos
- Voting Regressor
- Stacking

### 4. Tratamiento de Datos
- Análisis de outliers
- Tratamiento de desbalanceo (si existe)
- Feature selection

## 📚 Referencias

- [Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

**Autor**: Sistema de Análisis de Vinos  
**Última actualización**: 2024