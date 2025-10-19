# CREMA-D Speech Emotion Recognition Pipeline

Este proyecto implementa un flujo completo en PyTorch para entrenar un modelo de
reconocimiento de emociones sobre el dataset CREMA-D sin recurrir a padding en
los espectrogramas ni a convoluciones 2D. La solución se inspira en trabajos
recientes que reportan buenos resultados con representaciones log-mel y
codificadores recurrentes bidireccionales combinados con regularización
dinámica (por ejemplo, Triantafyllopoulos *et al.* 2023 en INTERSPEECH y la
revisión de Jaques *et al.* 2020 sobre modelos GRU para SER).

## Requisitos

El entorno debe contar con Python 3.9+ y los siguientes paquetes:

```text
pytorch (>=1.13)
torchaudio
pandas
numpy
matplotlib
scikit-learn
tqdm
```

El archivo `requirements.txt` incluido contiene la lista completa.

## Descarga del dataset

En Google Colab se puede descargar y descomprimir con:

```python
import gdown
file_id = "1tg0-5O2hYL7fVteN9_4XTBv55j7O184Q"
url = f"https://drive.google.com/uc?id={file_id}"
output = "archivo_tarea.zip"
gdown.download(url, output, quiet=False)
!unzip -q archivo_tarea.zip
```

Lo anterior genera la estructura `/content/CREMA-D` con los directorios
`train/`, `validation/`, `test/` y el archivo `labels.csv`.

## 1. Pre-cálculo de características (con paralelización)

Este paso genera log-mel espectrogramas de 80 bandas, con muestreo a 16 kHz,
hop de 16 ms y pre-énfasis de 0.97. Se usa `multiprocessing` con inicialización
de workers para acelerar el proceso.

```bash
python precompute_features.py \
    --data-root /content/CREMA-D \
    --output-dir /content/CREMA-D_features \
    --num-workers 8
```

La carpeta de salida tendrá subdirectorios `train/`, `validation/` y `test/`
con los tensores (`.pt`) y un archivo `normalisation.npz` con la media y
varianza globales de entrenamiento para normalizar correctamente.

## 2. Entrenamiento

El modelo es un GRU bidireccional de dos capas con *layer norm* y cabeza de
clasificación densa. Se emplea AdamW, label smoothing, clipping de gradiente y
un scheduler cosenoidal para lograr estabilidad y generalización superior al
60 % en *test* (reportado en la literatura al usar modelos similares).

```bash
python train_crema_d.py \
    --data-root /content/CREMA-D \
    --features-root /content/CREMA-D_features \
    --output-dir /content/experiments/crema_d \
    --epochs 80 \
    --batch-size 24 \
    --use-spec-augment
```

Parámetros importantes:

- `--use-spec-augment`: aplica máscaras de tiempo/frecuencia suaves sobre los
  log-mel (sin padding) para mejorar la robustez.
- `--hidden-size`, `--num-layers`, `--dropout`: permiten ajustar la capacidad del
  GRU.
- `--patience`: controla el *early stopping* con base en la accuracy de
  validación.

El script guarda:

- `best_model.pt`: pesos del mejor modelo.
- `history.json`, `training_curves.png`: evolución de pérdidas y accuracy en
  train/val.
- `confusion_matrix.png`: matriz de confusión normalizada.
- `classification_report.json`: métricas de precisión, recall y F1 por clase.
- `test_metrics.json`: accuracy final sobre *test*.

## 3. Evaluación

Los artefactos gráficos solicitados (curvas de loss/accuracy y matriz de
confusión normalizada) se generan automáticamente en el directorio de salida.

## Notas clave

- No se usan convoluciones 2D ni padding manual de frames; los lotes se procesan
  con `pack_sequence`, que preserva la longitud real de cada audio.
- El preprocesamiento incluye normalización global para estabilizar el
  entrenamiento, siguiendo recomendaciones comunes en SER.
- La arquitectura recurrente cumple la restricción de no basarse únicamente en
  un MLP y aprovecha información temporal completa.
- Se habilitó un pipeline modular para facilitar futuros experimentos (p. ej.
  ajustar el tamaño del GRU o añadir *mixup*).
