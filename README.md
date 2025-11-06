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
│
├── data/
│   └── wine.csv                    # Dataset
│
├── src/
│   ├── analysis/
│   │   ├── eda.py                  # Análisis exploratorio
│   │
│   ├── model/
│   │   └── logistic_regression.py       # Regresión Logística 3 clases
│   │
│   ├── config.py
│   └── screen.py
│
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
