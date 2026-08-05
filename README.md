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
