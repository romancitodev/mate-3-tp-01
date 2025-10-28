# TP-01 - Análisis de Precios de Juegos en Steam

## 👤 Integrantes del proyecto:
- Tomas Mesa
- Roman Fabris

## 🗃️ Dataset utilizado:
[steam](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data)

## ℹ️ Variables del dataset:
| Nombre | Descripción |
| --- | --- |
| appid | Identificador único de la aplicación en Steam. |
| name | Nombre del juego. |
| release_date | Fecha de lanzamiento del juego. |
| required_age | Edad mínima recomendada para jugar. |
| price | Precio actual en USD. |
| dlc_count | Cantidad de contenidos adicionales o DLCs. |
| detailed_description | Descripción completa y detallada del juego. |
| about_the_game | Descripción general del juego. |
| short_description | Descripción breve del juego. |
| reviews | Opiniones y comentarios de los usuarios (resumen). |
| header_image | Imagen principal o de portada del juego. |
| website | Página web oficial del juego. |
| support_url | Enlace al soporte del juego. |
| support_email | Correo electrónico de soporte del juego. |
| windows | Indica si está disponible en Windows. |
| mac | Indica si está disponible en Mac. |
| linux | Indica si está disponible en Linux. |
| metacritic_score | Puntuación en Metacritic. |
| metacritic_url | Enlace a la página de Metacritic del juego. |
| achievements | Cantidad de logros disponibles en el juego. |
| recommendations | Número de usuarios que recomiendan el juego. |
| notes | Advertencias sobre el contenido del juego (violencia, edad, etc.). |
| supported_languages | Lista de idiomas disponibles en texto. |
| full_audio_languages | Lista de idiomas disponibles en audio. |
| packages | Conjunto de juegos, DLC u otros ítems vendidos juntos. |
| developers | Desarrolladores del juego (separados por comas). |
| publishers | Publicadores del juego (separados por comas). |
| categories | Categorías del juego (separadas por comas). |
| genres | Géneros del juego (separados por comas). |
| screenshots | Imágenes o capturas de pantalla del juego. |
| movies | Videos o trailers del juego. |
| user_score | Puntuación general de los usuarios. |
| score_rank | Clasificación del juego según reseñas de usuarios. |
| positive | Cantidad de reseñas positivas. |
| negative | Cantidad de reseñas negativas. |
| estimated_owners | Número estimado de jugadores que compraron el juego. |
| average_playtime_forever | Promedio de tiempo jugado desde marzo de 2009 (minutos). |
| average_playtime_2weeks | Promedio de tiempo jugado en las últimas dos semanas (minutos). |
| median_playtime_forever | Mediana de tiempo jugado desde marzo de 2009 (minutos). |
| median_playtime_2weeks | Mediana de tiempo jugado en las últimas dos semanas (minutos). |
| discount | Porcentaje de descuento actual. |
| peak_ccu | Máximo de jugadores concurrentes ayer. |
| tags | Etiquetas del juego (separadas por comas). |
| pct_pos_total | Porcentaje de reseñas positivas de todos los tiempos. |
| num_reviews_total | Cantidad total de reseñas a lo largo del tiempo. |
| pct_pos_recent | Porcentaje de reseñas positivas recientes (últimos 30 días). |
| num_reviews_recent | Número de reseñas en los últimos 30 días. |

## 🎯 Objetivo del análisis
Predecir el precio de un juego en steam según sus características.

## 📥 Descargando
### Con pip convencional
```
pip install -r requirements.txt
```

### con uv:
```
uv sync
```
