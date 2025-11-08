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
## 📚 Estructura del Proyecto

```
mate-tp1/
├── data/
│   └── wine.csv                    # Dataset
├── src/
│   ├── analysis/
│   │   ├── eda.py                  # Análisis exploratorio
│   ├── model/
│   │   └── logistic_regression.py       # Regresión Logística 3 clases
│   ├── config.py
│   └── screen.py
├── main.py                         # Punto de entrada
├── requirements.txt                # Dependencias
└── README.md
```

## 🤖 Herramientas de IA Utilizada

**Anthropic Claude (Sonnet 4.5):**
- Roadmap y estructura del proyecto
- Investigación de mejores prácticas
- Optimización de visualizaciones
- Mejora de documentación del proyecto

---

## 🧠 ¿Por qué elegimos regresión logistica en vez de lineal?
Elegimos regresión logística en lugar de regresión lineal porque nuestro objetivo es clasificar la calidad del vino en categorías discretas (Bajo, Medio, Alto) en lugar de predecir un valor continuo. La regresión logística es más adecuada para problemas de clasificación, ya que modela la probabilidad de pertenencia a cada clase y utiliza una función sigmoide para mapear las predicciones a un rango entre 0 y 1. Esto permite asignar cada muestra a una categoría específica basándose en umbrales de probabilidad, lo cual no es posible con la regresión lineal que está diseñada para predecir valores continuos.

## 🤔 ¿Cómo sabemos que estamos yendo en la dirección correcta?
En base a las métricas de evaluación del modelo, como la precisión, el recall y la matriz de confusión, podemos determinar si nuestro modelo de regresión logística está funcionando correctamente. Si estas métricas muestran un buen desempeño en la clasificación de las categorías de calidad del vino, podemos concluir que estamos yendo en la dirección correcta. Además, realizamos validación cruzada para asegurarnos de que el modelo generaliza bien a datos no vistos.

## ✅ ¿Qué resultados nos esperamos?
Sabíamos que el alcohol era un factor muy importante, pero también entendimos que la acidez respecto de la densidad o el ácido cítrico por ejemplo también lo es y respecto de calidad es la cantidad de alcohol que tiene.
(el gráfico de matríz de correlación nos respalda).
Esperamos obtener un modelo que clasifique correctamente la calidad del vino en las categorías definidas (Bajo, Medio, Alto) con una precisión significativa. Además, esperamos identificar las características más influyentes que afectan la calidad del vino, lo que puede proporcionar información valiosa para los productores de vino. En términos de métricas.

## 💪 ¿Qué logramos?
Logramos implementar regresión logística que tiene un 72% de predicción para alta calidad, una precisión del 39% en calidad media y un 72% en baja calidad. (La calidad media se vio afectada por la cantidad total de muestras pero es porque el modelo es ligero).

En lineas generales logramos un 60% de precisión total en el modelo, lo cual es un buen resultado considerando la simplicidad del modelo y la naturaleza del dataset.
