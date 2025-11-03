# TP-01 - Predicción de calidad variada de vinos

## 👤 Integrantes del proyecto:
- Tomas Mesa
- Roman Fabris

## 🗃️ Dataset utilizado:
[Wine data](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

## ℹ️ Variables del dataset:
| Nombre | Descripción |
| --- | --- |
| fixed acidity | Amount of tartaric acid (g/dm³) |
| volatile acidity | Amount of acetic acid (g/dm³) |
| citric acid | Amount of citric acid (g/dm³) |
| residual sugar | Amount of sugar remaining after fermentation (g/dm³) |
| chlorides | Amount of salt (sodium chloride) in wine (g/dm³) |
| free sulfur dioxide | Free form of SO₂, prevents microbial growth and oxidation (mg/dm³) |
| total sulfur dioxide | Total SO₂, both free and bound forms (mg/dm³) |
| density | Density of wine (g/cm³) |
| pH | Acidity or basicity of wine (pH scale) |
| sulphates | Amount of potassium sulphate, contributes to wine stability (g/dm³) |
| alcohol | Alcohol content (% by volume) |
| quality | (score between 0 and 10) |


## 🎯 Objetivo del análisis
Predecir la calidad del vino clasificada en 3 categorías (Bajo/Medio/Alto) usando **Regresión Logística**.

## 📊 Modelo Implementado

### Regresión Logística - 3 Categorías

El problema se aborda como **clasificación multi-clase** con las siguientes categorías:

- **Bajo (3-5)**: Vinos de baja calidad
- **Medio (6)**: Vinos de calidad media
- **Alto (7-9)**: Vinos de alta calidad

### ¿Por qué 3 categorías?

✅ **Ventajas:**
- Simplifica el problema original de 7 clases (3-9) a 3 categorías
- Más interpretable para usuarios finales
- Balance entre simplicidad y granularidad
- Reduce el impacto del desbalanceo de clases

## 🔬 Técnicas Aplicadas

### 1. **StandardScaler** (Escalado de datos)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**¿Por qué?** Las variables tienen rangos muy diferentes:
- `density`: ~0.99 - 1.00
- `alcohol`: ~8 - 15
- `total sulfur dioxide`: ~6 - 289

StandardScaler normaliza todas las features a **media=0** y **std=1**, crucial para Regresión Logística.

### 2. **Estrategias de Balanceo de Clases**

El modelo prueba automáticamente 4 estrategias y selecciona la mejor:

1. **Sin balanceo** (baseline)
2. **class_weight='balanced'** (peso automático)
3. **class_weight manual** (Bajo x5, Medio x1, Alto x1)
4. **class_weight agresivo** (Bajo x10, Medio x1, Alto x1)

El sistema selecciona la estrategia que maximiza el **accuracy promedio por clase**, asegurando que todas las clases (especialmente la minoritaria "Bajo") tengan buen rendimiento.

### 3. **Cross-Validation**
- 5-Fold Cross-Validation para validación robusta
- Evita overfitting y da estimaciones más confiables

## 📥 Instalación de Dependencias

### Con pip convencional
```bash
pip install -r requirements.txt
```

### Con uv:
```bash
uv sync
```

## 🚀 Ejecución

```bash
python main.py
```

Esto ejecutará:
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Distribución de calidad y correlaciones
- ✅ Escalado con StandardScaler
- ✅ Comparación de 4 estrategias de balanceo
- ✅ Entrenamiento del mejor modelo
- ✅ Métricas detalladas (Accuracy, Precision, Recall, F1)
- ✅ Matriz de confusión
- ✅ Importancia de features
- ✅ Visualizaciones (4 gráficos)

## 📈 Performance Esperada

### Métricas Típicas

| Métrica | Valor Esperado |
|---------|----------------|
| **Accuracy Global** | 60-70% |
| **Accuracy Clase Bajo** | 30-50% (con balanceo) |
| **Accuracy Clase Medio** | 70-75% |
| **Accuracy Clase Alto** | 75-80% |
| **F1-Score (weighted)** | 0.60-0.68 |

### Interpretación

- El modelo clasifica correctamente ~65% de los vinos
- La clase "Bajo" es desafiante por tener pocas muestras (~7% del dataset)
- Las clases "Medio" y "Alto" tienen buen rendimiento (>70%)
- El balanceo automático mejora significativamente la clase minoritaria

## 📊 Visualizaciones Generadas

1. **Matriz de Confusión**: Muestra predicciones correctas e incorrectas por clase
2. **Distribución de Predicciones**: Cantidad de vinos clasificados en cada categoría
3. **Accuracy por Clase**: Performance individual de cada categoría
4. **Cross-Validation Scores**: Consistencia del modelo en diferentes folds

## 🔍 Análisis de Features

El modelo muestra la **importancia de cada variable** según sus coeficientes:

**Top features típicas:**
- `alcohol`: Mayor contenido alcohólico → Mejor calidad
- `volatile acidity`: Mayor acidez volátil → Peor calidad
- `sulphates`: Más sulfatos → Mejor calidad
- `density`: Menor densidad → Mejor calidad
- `citric acid`: Más ácido cítrico → Mejor calidad

## 💡 Ventajas del Modelo

✅ **Interpretabilidad**: Coeficientes claros que explican las decisiones  
✅ **Rapidez**: Entrenamiento en <1 segundo  
✅ **Simplicidad**: Fácil de entender y mantener  
✅ **Balanceo Automático**: Compensa clases desbalanceadas  
✅ **Probabilidades**: Proporciona confianza en predicciones  

## 📚 Estructura del Proyecto

```
mate-tp1/
│
├── data/
│   └── wine.csv                    # Dataset
│
├── src/
│   ├── analysis/
│   │   ├── eda.py                  # Análisis exploratorio
│   │   └── classification.py       # Regresión Logística 3 clases
│   ├── config.py
│   └── screen.py
│
├── main.py                         # Punto de entrada
├── requirements.txt                # Dependencias
└── README.md
```

## 🎓 Conclusiones

1. **Regresión Logística** es efectiva para clasificar vinos en 3 categorías
2. El **escalado con StandardScaler** es crítico para el rendimiento
3. Las **estrategias de balanceo** mejoran significativamente la clase minoritaria
4. El modelo es **interpretable** y puede explicar sus decisiones
5. Las variables químicas más importantes son: **alcohol**, **acidez volátil** y **sulfatos**

## 🔧 Próximos Pasos Sugeridos

- Optimizar hiperparámetros (C, penalty) con GridSearchCV
- Probar feature engineering (interacciones, polinomios)
- Recolectar más datos de vinos de baja calidad
- Experimentar con diferentes umbrales de categorización
- Comparar con otros modelos (Random Forest, SVM)

---

**Desarrollado como parte del TP-01 de Matemática para el Aprendizaje Automático**