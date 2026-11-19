#  Super Fast Delivery — Predicción del tiempo de entrega de pedidos (Rappi)

Proyecto de **Inteligencia Artificial** (Universidad Industrial de Santander) que estima el **tiempo de entrega de un pedido en minutos** (`Time_taken_min`) usando únicamente información disponible antes del envío. El trabajo combina análisis exploratorio (EDA), aprendizaje supervisado, reducción de dimensionalidad (PCA) y aprendizaje no supervisado (clustering).

---

##  Video de presentación

 **[https://drive.google.com/file/d/1mwK-jojOiz8UXUJg07LAoyqzkGifWviD/view?usp=sharing]**

---

##  Integrantes

| Nombre | Codigo/|
|---|---|
| _[Diego Alejandro Sanabria Perez]_ | _[2231865]_ |
| _[Erick Joan Lizarazo Pardo]_ | _[2231875]_ |
| _[Samuel David Penilla Ramirez]_ | _[2231893]_ |
| _[Eimy Natalia Montaño Ortiz]_ | _[2231881]_ |

---

##  Datos utilizados

### Fuente

| Campo | Detalle |
|---|---|
| **Dataset** | Food Delivery Time Prediction Dataset |
| **Plataforma** | Kaggle |
| **Enlace** | https://www.kaggle.com/datasets/dharmendrapandit12/food-delivery-time-prediction-dataset |
| **Identificador (kagglehub)** | `dharmendrapandit12/food-delivery-time-prediction-dataset` |
| **Archivo usado** | `Food_Delivery_Time_Prediction.csv` |
| **Tamaño** | 50.000 filas × 24 columnas |
| **Variable objetivo** | `Time_taken_min` (minutos) |

>  **Nota sobre los datos:** no son datos internos de Rappi. Se trata de un dataset público de domicilios de comida que usamos como caso de estudio y que, por su comportamiento, parece **sintético** (se discute como limitación en las conclusiones del notebook).

### Ubicación en este repositorio

- **Ruta esperada:** `data/Food_Delivery_Time_Prediction.csv` _(ajustar si el archivo se sube al repositorio; si no se sube, dejar la carpeta vacía o con un `.gitkeep`)_.
- El CSV **no es obligatorio en el repositorio**: el notebook lo descarga automáticamente si no lo encuentra (ver abajo).

### Cómo obtener los datos

**Opción A — Automática (recomendada).** No requiere hacer nada: la celda de carga del notebook busca `Food_Delivery_Time_Prediction.csv` en el directorio de trabajo y, si no existe, lo descarga con `kagglehub`:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "dharmendrapandit12/food-delivery-time-prediction-dataset",
    "Food_Delivery_Time_Prediction.csv",
)
```

**Opción B — Descarga manual.**
1. Abrir el enlace de Kaggle de arriba (puede requerir iniciar sesión gratuita).
2. Pulsar **Download** y descomprimir el `.zip`.
3. Copiar `Food_Delivery_Time_Prediction.csv` junto al notebook (o dentro de `data/`).

**Opción C — Kaggle CLI.**
```bash
pip install kaggle
kaggle datasets download -d dharmendrapandit12/food-delivery-time-prediction-dataset --unzip
```
> Requiere configurar tu token de API de Kaggle (`~/.kaggle/kaggle.json`).

### Archivos donde se procesan los datos

| Archivo | Qué hace con los datos |
|---|---|
| `Proyecto_Rappi_IA.ipynb` | Todo el flujo: carga, limpieza, EDA, modelos supervisados, PCA y clustering. |

### Descripción de las variables

Las 24 columnas originales incluyen, entre otras: `Order_ID`, `Order_Date`, `Order_Hour`, `Day_of_Week`, `Is_Weekend`, `Is_Festival`, `Weather`, `Pickup_Zone`, `Dropoff_Zone`, `Vehicle_Type`, `Rider_Experience_Years`, `Rider_Rating`, `Restaurant_Rating`, `Cuisine_Type`, `Order_Items`, `Restaurant_Load`, `Preparation_Time_Min`, `Road_Distance_km`, `Delivery_Distance_Category`, `Traffic_Level`, `Number_of_Signals`, `Average_Speed_kmph` y la variable objetivo `Time_taken_min`.

**Decisiones de preparación:**

- Se descartan identificadores (`Order_ID`) y la fecha cruda (`Order_Date`).
- Se **excluye `Average_Speed_kmph`** por **fuga de información** (la velocidad media se calcula con el tiempo real de entrega).
- Se eliminan columnas auxiliares del EDA construidas con el objetivo o con todo el dataset (`target_time`, `distance_group`, cuartiles, categorías de tiempo).
- Las variables categóricas se codifican con *one-hot encoding*, lo que lleva de 24 columnas originales a **48 variables predictoras**.

---

##  Estructura del proyecto

```
.
├── README.md
├── Proyecto_Rappi_IA.ipynb          # Notebook principal (todo el análisis)
├── data/
│   └── Food_Delivery_Time_Prediction.csv   # (opcional) ver sección de datos
└── requirements.txt                 # (opcional) dependencias
```

### Contenido del notebook

1. **Carga y revisión inicial** — lectura del CSV, tipos de datos, calidad y limpieza.
2. **Análisis de la variable objetivo** — distribución (sesgo a la derecha, hasta 180 min), Q-Q plot, outliers (IQR y z-score).
3. **EDA y ranking de variables** — correlaciones y asociaciones categóricas; relación con distancia, tráfico, clima, vehículo y prioridad.
4. **Aprendizaje supervisado** — Decision Tree, Random Forest, SVR y red neuronal densa (Keras), con `RandomizedSearchCV` y validación cruzada K-Fold (k=5).
5. **Reducción de dimensionalidad (PCA)** — scree plot, varianza acumulada, elección del número de componentes y *loadings*; comparación de los mismos modelos con y sin PCA.
6. **Aprendizaje no supervisado** — K-Means y clustering aglomerativo, métricas internas (silueta, Davies-Bouldin, Calinski-Harabasz) y externas (ARI, NMI), visualización con PCA y t-SNE.
7. **Conclusiones** — utilidad de cada técnica, limitaciones, mejoras futuras y conexión con los conceptos del curso.

---

##  Resultados principales

Evaluación en el conjunto de prueba (partición única 80/20, `shuffle=True`, `random_state=42`). Errores en minutos.

### Modelos supervisados, con y sin PCA

| Modelo | MAE sin PCA | MAE con PCA (42 comp.) | RMSE sin PCA | RMSE con PCA | R² sin PCA | R² con PCA |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 4,07 | 9,98 | 5,55 | 15,62 | 0,98 | 0,81 |
| Random Forest | 3,05 | 6,96 | 4,04 | 10,60 | 0,99 | 0,91 |
| SVR | 4,39 | 5,88 | 6,82 | 10,29 | 0,96 | 0,92 |
| **Red neuronal (Keras)** | **2,74** | 3,12 | **3,55** | 4,03 | **0,99** | 0,99 |

- La **red neuronal densa** fue el mejor modelo (MAE ≈ 2,7 min).
- **PCA no mejoró el desempeño:** empeoró a todos los modelos; el efecto fue menor en la red neuronal. Con 48 variables y 40.000 filas, la dimensionalidad no es un problema grave y la varianza está repartida entre muchos componentes.

### Clustering (K-Means, k = 4, espacio PCA)

- Silueta ≈ 0,07 y Davies-Bouldin ≈ 3,49: **estructura de clusters débil**.
- Los clusters se alinean con el nivel de tráfico (NMI ≈ 0,47), pero casi no con la categoría de tiempo (ARI ≈ 0,06) y explican solo ≈ 6 % de la varianza del tiempo (η²).

> Los números exactos pueden variar levemente según el entorno y las versiones de las librerías.

---

##  Cómo ejecutar el proyecto

### Opción 1 — Google Colab (recomendada)

1. Subir `Proyecto_Rappi_IA.ipynb` a [Google Colab](https://colab.research.google.com/).
2. Ejecutar las celdas en orden; el dataset se descarga solo con `kagglehub`.
3. Tiempo aproximado de la fase de PCA y clustering: **20–30 min en CPU**. Para una prueba rápida, poner `MODO_RAPIDO = True` en la primera celda de esa fase (esas cifras **no** deben usarse en el informe).

### Opción 2 — Entorno local

```bash
git clone <URL-DEL-REPOSITORIO>
cd <NOMBRE-DEL-REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow kagglehub scikeras jupyter

jupyter notebook Proyecto_Rappi_IA.ipynb
```

**Librerías principales:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `tensorflow` / `keras`, `kagglehub`, `scikeras`.

---

##  Limitaciones

- Dataset aparentemente sintético; los resultados no son directamente extrapolables a datos reales de Rappi.
- La codificación *one-hot* no es ideal para PCA ni para K-Means.
- SVR se entrenó con una submuestra por costo computacional.
- Los errores siguen siendo mayores en entregas extremas (> 120 min).
- No se hizo validación temporal (entrenar con el pasado y probar con el futuro).

##  Mejoras futuras

Ingeniería de variables (interacciones distancia × tráfico, hora pico), codificaciones alternativas para categóricas, modelos de *boosting* (XGBoost, LightGBM, CatBoost), validación temporal, intervalos de predicción y validación con datos reales.

---

##  Licencia y créditos

- Dataset: autor `dharmendrapandit12` en Kaggle; consultar la licencia indicada en la página del dataset.
- Proyecto académico — Inteligencia Artificial, Universidad Industrial de Santander.
