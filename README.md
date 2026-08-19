# Proyecto NLP - Data Science III

Sistema comparativo de clasificación de noticias mediante NLP: baseline clásico (TF-IDF + Regresión Logística) vs. Transformer ajustado con LoRA, sobre el corpus AG News.

Este repositorio contiene la implementación correspondiente a las entregas de la cursada de Data Science III. El diseño arquitectónico completo del proyecto está documentado en `docs/00_documento_arquitectura/` (versión congelada) y su adaptación al alcance de la cursada en `docs/01_plan_implementacion/`.

## Entrega 1 — Pipeline base con PyTorch

Notebook: `notebooks/01_pipeline_base_pytorch.ipynb`

### Objetivo

Construir y validar la infraestructura técnica base del proyecto: configuración del entorno, reproducibilidad, y un ciclo de entrenamiento/validación funcional en PyTorch, sobre un dataset clásico (Iris) independiente del corpus AG News.

### Entorno

- **Versión de PyTorch:** `<completar con el resultado de print(torch.__version__)>`
- **Dispositivo:** detección automática (`cuda` → `mps` → `cpu`). En el entorno de desarrollo se ejecutó sobre `cpu`.
- **Semilla:** `SEED = 42`, fijada en `random`, `numpy`, `torch` y `torch.cuda` para asegurar reproducibilidad.

### Configuración del entrenamiento

| Parámetro | Valor |
|---|---|
| Dataset | Iris (`sklearn.datasets.load_iris`) |
| Split | 80% train+val / 20% test, con 20% del train reservado para validación (96 train / 24 val / 30 test), estratificado |
| Estandarización | `StandardScaler`, ajustado únicamente sobre train |
| Arquitectura | `nn.Sequential(Linear(4,16), ReLU, Linear(16,3))` |
| Función de pérdida | `CrossEntropyLoss` |
| Optimizador | `Adam`, `learning_rate = 0.001` |
| Batch size | 16 |
| Épocas | 200 |

### Resultados

- **Test loss:** 0.1296
- **Test accuracy:** 96.67%
- **Matriz de confusión:** clasificación perfecta de Setosa y Virginica; 1 de 10 observaciones de Versicolor clasificada incorrectamente como Virginica.

### Interpretación de la curva de pérdida

Tanto la pérdida de entrenamiento como la de validación descienden de forma sostenida durante las primeras ~100 épocas y luego se estabilizan, alcanzando convergencia sin señales de sobreajuste (la curva de validación no diverge respecto de la de entrenamiento). El resultado final (96.67% de accuracy en test) es consistente con lo esperable para un dataset linealmente casi separable como Iris.

### Artefactos generados

- `outputs/models/modelo_iris.pth` — pesos del modelo entrenado.
- `outputs/metrics/history.json` — historial de pérdida y accuracy por época (train/val).
- `outputs/figures/loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png` — gráficos de la corrida.

## Entrega 2 — Análisis Exploratorio y Preprocesamiento del Corpus

Notebook: `notebooks/02_eda_y_preprocesamiento.ipynb`

### Objetivo

Realizar un análisis exploratorio del corpus AG News y desarrollar un pipeline de preprocesamiento orientado a la preparación de los datos para tareas de clasificación automática de noticias: características generales del dataset, distribución de clases y longitud de los documentos, seguido de limpieza, normalización y lematización del texto.

### Pipeline de preprocesamiento

1. Conversión de todos los documentos a minúsculas.
2. Eliminación de URLs y caracteres no alfabéticos mediante expresiones regulares.
3. Tokenización utilizando el modelo `en_core_web_sm` de spaCy.
4. Eliminación de signos de puntuación y stopwords.
5. Lematización de los términos.
6. Decodificación de entidades HTML escapadas y posterior eliminación de etiquetas HTML.
7. Eliminación de un conjunto específico de términos considerados ruido (agencias de noticias y otros artefactos técnicos, identificado durante el EDA). El conjunto completo se detalla en el informe de `reports/entrega_02`.

### Corpus

El corpus utilizado contiene 8.000 documentos de entrenamiento distribuidos uniformemente entre las cuatro categorías temáticas (Business, Sci_Tech, Sports y World), con 2.000 documentos por clase, y 2.000 documentos de prueba (500 por clase). La longitud de los documentos se concentra alrededor de una media cercana a 38 palabras, con un percentil 95 de 53 palabras.

### Top n-gramas (extracto)

| Bigrama | Frecuencia | Trigrama | Frecuencia |
|---|---|---|---|
| new york | 419 | quote profile research | 69 |
| oil price | 197 | boston red sox | 37 |
| united states | 167 | new york stock | 37 |
| prime minister | 139 | high oil price | 27 |
| official say | 115 | canadian press canadian | 26 |

Los n-gramas más frecuentes representan principalmente entidades geográficas, económicas y políticas, coherentes con la temática periodística del corpus.

### Corrección posterior: ruido HTML/CSS en el preprocesamiento

La validación inicial de `preprocess_text()` no tuvo la robustez necesaria: el análisis de trigramas reveló términos residuales de formato HTML/CSS (por ejemplo, "ms sans serif", "verdana ms sans"), provenientes de etiquetas HTML almacenadas mediante entidades escapadas que la limpieza original no reconocía como tags completos. El bigrama top-20 no se vio afectado — el problema se concentró exclusivamente en el top-20 de trigramas.

Se corrigió el pipeline incorporando la decodificación de entidades HTML *antes* de eliminar las etiquetas de formato, y se volvió a ejecutar el Notebook 02 completo. Los nuevos trigramas confirmaron que los residuos HTML/CSS dejaron de aparecer. El corpus corregido se exportó con un nombre distinto al histórico (`data/processed/ag_news_train_processed_limpio.csv` / `ag_news_test_processed_limpio.csv`), para no sobrescribir los archivos que ya habían sido consumidos por las Entregas 3 y 4.

Se decidió **no** reentrenar retrospectivamente los modelos de las Entregas 3 y 4 sobre el corpus corregido, para evitar sesgar los resultados retrospectivamente en busca de un desempeño más favorable a posteriori. Ambas ramas (TF-IDF + Regresión Logística y DistilBERT + LoRA) siguen entrenadas y evaluadas sobre la misma versión del corpus original, preservando la comparabilidad entre ellas. Esta limitación queda documentada como restricción metodológica y como base de hipótesis para una futura expansión del proyecto (ver `docs/00_documento_arquitectura/`).

### Artefactos generados

- `data/processed/ag_news_train_processed.csv`, `data/processed/ag_news_test_processed.csv` — corpus histórico, utilizado por las Entregas 3 y 4.
- `data/processed/ag_news_train_processed_limpio.csv`, `data/processed/ag_news_test_processed_limpio.csv` — corpus corregido (limpieza HTML/CSS robusta), no consumido todavía por ningún modelo.
- `outputs/tables/top_50_palabras.csv`, `top_20_bigramas.csv`, `top_20_trigramas.csv` — tablas recalculadas sobre el corpus corregido.
- `outputs/figures/` — distribución de clases, distribución de longitud de documentos.
- `reports/entrega_02/EDA_NLP_FernándezAmilcar.PDF` — informe técnico de la entrega, con la comparación completa de n-gramas antes/después de la corrección.

## Entrega 3 — Baseline TF-IDF + Regresión Logística

Notebook: `notebooks/03_baseline_tfidf.ipynb`

### Objetivo

Construir el baseline clásico de clasificación temática del proyecto, sobre el corpus AG News preprocesado en la Entrega 2, utilizando TF-IDF como representación vectorial y Regresión Logística como clasificador.

### Justificación del modelo

El baseline se implementó con **Regresión Logística sobre TF-IDF**, decisión congelada desde el diseño arquitectónico (Documento 00, Anexo A) por los siguientes motivos:

- Es uno de los modelos desarrollados en el material de la plataforma.
- Constituye un baseline ampliamente utilizado en clasificación de texto.
- Ofrece buen equilibrio entre desempeño, interpretabilidad y costo computacional frente a alternativas como Naive Bayes o SVM, que quedaron documentadas como alternativas de referencia pero no se implementaron.

### Selección de la configuración del vectorizador

Se compararon tres configuraciones de `TfidfVectorizer` (`max_features` y `ngram_range`) entrenando el mismo clasificador sobre cada una. Para evitar sesgar la evaluación final, la comparación se hizo sobre un conjunto de **validación** separado del train (20%, estratificado) — el test oficial permaneció aislado hasta la evaluación final.

| Configuración | max_features | ngram_range | Accuracy (validación) |
|---|---|---|---|
| config_1 | 5.000 | (1,1) | 0.8906 |
| config_2 | 10.000 | (1,1) | **0.8925** |
| config_3 | 10.000 | (1,2) | 0.8919 |

Configuración adoptada: **`max_features=10000`, `ngram_range=(1,1)`**. El vectorizador y el modelo definitivos se reentrenaron sobre el conjunto de entrenamiento completo (8.000 documentos) antes de la evaluación final sobre test.

### Resultados sobre test

| Métrica | Valor |
|---|---|
| Accuracy | 0.8930 |
| F1-Score macro | 0.8930 |

| Categoría | Precision | Recall | F1-Score |
|---|---|---|---|
| Business | 0.8428 | 0.8580 | 0.8503 |
| Sci_Tech | 0.8745 | 0.8780 | 0.8762 |
| Sports | 0.9487 | 0.9620 | 0.9553 |
| World | 0.9066 | 0.8740 | 0.8900 |

### Análisis preliminar (matriz de confusión)

`Sports` es la categoría mejor diferenciada (481/500 correctas). La principal dificultad del modelo se concentra entre `Business` y `Sci_Tech`: 40 documentos de `Business` se clasificaron como `Sci_Tech`, y 42 de `Sci_Tech` como `Business` — categorías con vocabulario económico/tecnológico solapado (empresas, mercados, productos). Se observa también confusión menor entre `World` y `Business` (31 casos).

### Artefactos generados

- `outputs/figures/matriz_confusion_tfidf_logistica.png`
- `outputs/tables/metricas_baseline_tfidf_logistica.csv`

### Limitación metodológica documentada

Seleccionar la configuración del vectorizador usando el propio test habría introducido un sesgo optimista en la métrica final; por eso la selección se hizo sobre un conjunto de validación separado del train, dejando el test reservado exclusivamente para la evaluación del modelo ya definido.

Este modelo fue entrenado sobre el corpus histórico de la Entrega 2 (`ag_news_*_processed.csv`), que contenía ruido técnico HTML/CSS no filtrado — ver "Corrección posterior" en la sección de Entrega 2 para el detalle completo y la justificación de por qué no se reentrenó sobre el corpus corregido.

## Entrega 4 — Transformer + LoRA

Notebook: `notebooks/04_transformer_lora.ipynb`

### Objetivo

Implementar la rama avanzada de clasificación temática mediante fine-tuning eficiente (PEFT) de un modelo Transformer preentrenado sobre el mismo corpus AG News utilizado en las etapas anteriores, y contrastar su desempeño contra el baseline clásico de la Entrega 3 (TF-IDF + Regresión Logística), evaluados sobre el mismo conjunto de prueba.

### Selección del modelo y configuración LoRA

Se seleccionó **DistilBERT** (`distilbert-base-uncased`) como modelo base, por ser compatible con un corpus en inglés y ofrecer una arquitectura más liviana que BERT, adecuada para el ajuste mediante LoRA sobre una GPU Tesla T4 de Google Colab. Sobre este modelo se aplicó LoRA mediante la librería PEFT, congelando el modelo base e incorporando matrices de bajo rango sobre los módulos de atención.

| Parámetro | Valor |
|---|---|
| Modelo base | distilbert-base-uncased |
| r | 8 |
| lora_alpha | 16 |
| lora_dropout | 0,1 |
| Módulos objetivo | q_lin, v_lin |

`r=8` se adoptó como rango intermedio para dotar de capacidad de adaptación sin incrementar innecesariamente los parámetros entrenables en una tarea de clasificación de 4 clases; `lora_alpha=16` (el doble del rango) mantiene una escala moderada de actualización de los adaptadores.

La configuración resultante dejó **741.124 parámetros entrenables sobre un total de 67.697.672 (1,0948 %)**, muy por debajo del límite del 3 % sugerido por la consigna.

### Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0,0 (valor por defecto de `TrainingArguments`) |
| Criterio de selección | F1-Score macro |
| Hardware | GPU NVIDIA Tesla T4 (Google Colab) |
| Tiempo total de entrenamiento | 78,58 s (1,31 min) |

| Época | Training Loss | Validation Loss | Accuracy (val) | F1 macro (val) |
|---|---|---|---|---|
| 1 | 0,8829 | 0,4213 | 0,8663 | 0,8651 |
| 2 | 0,3903 | 0,3599 | 0,8781 | 0,8774 |
| 3 | 0,3639 | 0,3535 | 0,8781 | 0,8774 |

Ambas pérdidas descienden de forma sostenida sin señales de sobreajuste; la mejora se concentra entre la época 1 y 2, y se estabiliza en la 3.

### Resultados sobre test

| Métrica | Valor |
|---|---|
| Accuracy | 0,8800 |
| Precision macro | 0,8813 |
| Recall macro | 0,8800 |
| F1-Score macro | 0,8804 |

| Categoría | Precision | Recall | F1-Score |
|---|---|---|---|
| World | 0,8871 | 0,8800 | 0,8835 |
| Sports | 0,9658 | 0,9600 | 0,9629 |
| Business | 0,7955 | 0,8400 | 0,8171 |
| Sci_Tech | 0,8768 | 0,8400 | 0,8580 |

### Análisis preliminar (matriz de confusión)

`Sports` es la categoría mejor diferenciada (480/500 correctas). La principal dificultad se concentra entre `Business` y `Sci_Tech`: 46 documentos de `Business` se clasificaron como `Sci_Tech`, y 58 de `Sci_Tech` como `Business`. También se observa confusión menor entre `World` y `Business` (39 casos).

### Comparación con el baseline clásico (Entrega 3)

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---|---|---|
| TF-IDF + Regresión Logística | 0,8930 | 0,8932 | 0,8930 | 0,8930 |
| DistilBERT + LoRA | 0,8800 | 0,8813 | 0,8800 | 0,8804 |

El baseline clásico obtuvo un rendimiento global levemente superior (+0,0130 accuracy, +0,0126 F1 macro). Por categoría, DistilBERT + LoRA solo superó al baseline en `Sports` (+0,0076 F1); en `World`, `Business` y `Sci_Tech` el modelo clásico fue superior, con la mayor brecha en `Business` (-0,0332 F1).

### Conclusión técnica

En estas condiciones, el incremento de complejidad del Transformer no se tradujo en una mejora de desempeño frente al baseline clásico. Una hipótesis metodológica es que DistilBERT recibió el corpus `text_processed` (ya lematizado y sin stopwords), pensado originalmente para TF-IDF, lo que pudo reducir información contextual que un Transformer preentrenado aprovecha mejor sobre texto más cercano al original. Queda como extensión pendiente comparar ambos enfoques bajo las mismas condiciones pero con texto menos normalizado como entrada al Transformer.

### Artefactos generados

- `outputs/figures/matriz_confusion_transformer_lora.png`
- `outputs/tables/metricas_transformer_lora.csv`
- `outputs/tables/predicciones_transformer_lora.csv`
- `reports/entrega_04/Fernández_Amilcar_Checkpoint_NLP3.PDF` — informe técnico de la pre-entrega.

### Limitación metodológica documentada

Al igual que el baseline de la Entrega 3, este modelo fue entrenado y evaluado sobre el corpus histórico de la Entrega 2 (`ag_news_*_processed.csv`) — ver "Corrección posterior" en la sección de Entrega 2. Como ambas ramas (TF-IDF y Transformer) comparten la misma versión del corpus, la comparación entre ellas sigue siendo válida dentro de las condiciones originales del experimento; lo que queda como pregunta abierta es si el resultado de esta comparación se sostendría sobre el corpus corregido, no la comparación en sí.

## Notebook 05 — Comparación final de modelos

Notebook: `notebooks/05_comparacion_final.ipynb`

### Objetivo

Consolidar en un único análisis la comparación entre las dos ramas de clasificación desarrolladas en las Entregas 3 y 4 (TF-IDF + Regresión Logística vs. DistilBERT + LoRA), verificando primero la comparabilidad de ambas evaluaciones y sintetizando después los resultados en tablas y figuras unificadas.

### Verificación de comparabilidad

Antes de comparar métricas, se verificó que ambas ramas fueron evaluadas sobre el mismo conjunto de test (2.000 documentos, 500 por clase) y sobre la misma versión del corpus (histórico, `ag_news_*_processed.csv` — ver "Corrección posterior" en la sección de Entrega 2). Las métricas por clase del Transformer se recalcularon con `classification_report()` sobre `predicciones_transformer_lora.csv` en lugar de reutilizar valores ya reportados en el PDF de la Entrega 4, para evitar errores de transcripción y asegurar reproducibilidad.

### Resultados comparativos

| Métrica | TF-IDF + Regresión Logística | DistilBERT + LoRA |
|---|---|---|
| Accuracy | 0,8930 | 0,8800 |
| Precision macro | 0,8932 | 0,8813 |
| Recall macro | 0,8930 | 0,8800 |
| F1-Score macro | 0,8930 | 0,8804 |

`Sports` es la categoría mejor diferenciada en ambos modelos (481/500 y 480/500 respectivamente). La principal dificultad se concentra, en los dos casos, entre `Business` y `Sci_Tech`, con una confusión algo mayor en el Transformer (46 y 58 documentos cruzados entre ambas clases, frente a 40 y 42 en el baseline).

### Análisis de desacuerdos — limitación documentada

No fue posible construir un análisis de desacuerdos a nivel de documento individual entre ambos modelos, porque el Notebook 03 (baseline) nunca persistió sus predicciones por documento — solo las métricas agregadas. Esta limitación queda registrada como requisito para notebooks futuros: persistir siempre las predicciones crudas, no solo las métricas derivadas.

### Comparación de costo computacional — limitación documentada

Por el mismo motivo, tampoco es posible comparar tiempos de entrenamiento ni cantidad de parámetros entre ambas ramas de forma simétrica: el Notebook 03 no persistió esa información. Del Transformer se cuenta con el dato completo: 78,58 s de entrenamiento y 741.124 parámetros entrenables sobre un total de 67.697.672 (1,0948 %).

### Conclusión técnica

En las condiciones evaluadas, el baseline clásico (TF-IDF + Regresión Logística) obtuvo un desempeño global levemente superior al Transformer ajustado con LoRA. El resultado es consistente con la hipótesis ya registrada en la Entrega 4: el corpus de entrada (`text_processed`, lematizado y sin stopwords) favorece a un modelo de bolsa de palabras y le resta al Transformer parte del contexto que un modelo preentrenado aprovecha mejor. Este hallazgo se incorpora como base de dos hipótesis de segunda ronda para la expansión del proyecto (Documento 00): evaluar el Transformer sobre el corpus sin normalización lingüística, y evaluar el impacto de la limpieza HTML/CSS robusta (Entrega 2) sobre ambas ramas.

### Artefactos generados

- `outputs/tables/metricas_clases_unificadas.csv`, `outputs/tables/metricas_globales_unificadas.csv`
- `outputs/figures/comparacion_metricas_globales.png`, `comparacion_precision_por_categoria.png`, `comparacion_recall_por_categoria.png`, `comparacion_f1_por_categoria.png`

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del repositorio

```
proyecto_nlp_ds3/
├── data/               # raw, interim, processed
├── docs/               # documentación arquitectónica del proyecto
├── notebooks/          # notebooks numerados por etapa
├── outputs/            # figures, metrics, models, logs
├── reports/            # entregables por checkpoint
└── src/                # módulos reutilizables
```
