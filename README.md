# TP-01 - Análisis de Precios de Juegos en Steam

## 👤 Integrantes del proyecto:
- Tomas Mesa
- Roman Fabris

## 🗃️ Dataset utilizado:
[steam](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data)

## ℹ️ Variables del dataset:
| Nombre | Descripción | Se toma en cuenta |
| --- | --- | --- |
| appid | Identificador único de la aplicación en Steam. | Si |
| name | Nombre del juego. | No |
| release_date | Fecha de lanzamiento del juego. | Si |
| required_age | Edad mínima recomendada para jugar. | Si |
| price | Precio actual en USD. | Si |
| dlc_count | Cantidad de contenidos adicionales o DLCs. | Si |
| detailed_description | Descripción completa y detallada del juego. | No |
| about_the_game | Descripción general del juego. | No |
| short_description | Descripción breve del juego. | No |
| reviews | Opiniones y comentarios de los usuarios (resumen). | No |
| header_image | Imagen principal o de portada del juego. | No |
| website | Página web oficial del juego. | Si (mapeo a bool) |
| support_url | Enlace al soporte del juego. | No |
| support_email | Correo electrónico de soporte del juego. | No |
| windows | Indica si está disponible en Windows. | Si |
| mac | Indica si está disponible en Mac. | Si |
| linux | Indica si está disponible en Linux. | Si |
| metacritic_score | Puntuación en Metacritic. | Si |
| metacritic_url | Enlace a la página de Metacritic del juego. | No |
| achievements | Cantidad de logros disponibles en el juego. | Si (mapeo a int) |
| recommendations | Número de usuarios que recomiendan el juego. | Si (mapeo a int) |
| notes | Advertencias sobre el contenido del juego (violencia, edad, etc.). | No |
| supported_languages | Lista de idiomas disponibles en texto. | Si (mapeo a int) |
| full_audio_languages | Lista de idiomas disponibles en audio. | No |
| packages | Conjunto de juegos, DLC u otros ítems vendidos juntos. | Si (mapeo a int) |
| developers | Desarrolladores del juego (separados por comas). | Si |
| publishers | Publicadores del juego (separados por comas). | Si |
| categories | Categorías del juego (separadas por comas). | Si |
| genres | Géneros del juego (separados por comas). | Si |
| screenshots | Imágenes o capturas de pantalla del juego. | No |
| movies | Videos o trailers del juego. | No |
| user_score | Puntuación general de los usuarios. | Si |
| score_rank | Clasificación del juego según reseñas de usuarios. | Si |
| positive | Cantidad de reseñas positivas. | Si |
| negative | Cantidad de reseñas negativas. | Si |
| estimated_owners | Número estimado de jugadores que compraron el juego. | Si |
| average_playtime_forever | Promedio de tiempo jugado desde marzo de 2009 (minutos). | Si |
| average_playtime_2weeks | Promedio de tiempo jugado en las últimas dos semanas (minutos). | No |
| median_playtime_forever | Mediana de tiempo jugado desde marzo de 2009 (minutos). | No |
| median_playtime_2weeks | Mediana de tiempo jugado en las últimas dos semanas (minutos). | No |
| discount | Porcentaje de descuento actual. | Si |
| peak_ccu | Máximo de jugadores concurrentes ayer. | No |
| tags | Etiquetas del juego (separadas por comas). | Si |
| pct_pos_total | Porcentaje de reseñas positivas de todos los tiempos. | Si |
| num_reviews_total | Cantidad total de reseñas a lo largo del tiempo. | Si |
| pct_pos_recent | Porcentaje de reseñas positivas recientes (últimos 30 días). | Si |
| num_reviews_recent | Número de reseñas en los últimos 30 días. | Si |

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
