# Proyecto NLP - Data Science III

Sistema comparativo de clasificación de noticias mediante NLP: baseline clásico (TF-IDF + Regresión Logística) vs. Transformer ajustado con LoRA, sobre el corpus AG News.

Este repositorio contiene la implementación correspondiente a las entregas de la cursada de Data Science III. El diseño arquitectónico completo del proyecto está documentado en `docs/00_documento_arquitectura/` (versión congelada) y su adaptación al alcance de la cursada en `docs/01_plan_implementacion/`.

## Entrega 1 — Pipeline base con PyTorch

Notebook: `notebooks/01_pipeline_base_pytorch.ipynb`

### Objetivo

Construir y validar la infraestructura técnica base del proyecto: configuración del entorno, reproducibilidad, y un ciclo de entrenamiento/validación funcional en PyTorch, sobre un dataset clásico (Iris) independiente del corpus AG News.

### Entorno

- **Versión de PyTorch:** 2.13.0+cpu
- **Versión de Python:** 3.11.15
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
