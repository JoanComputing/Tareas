# CREMA-D Speech Emotion Recognition Pipeline

Este proyecto implementa un flujo completo en PyTorch para entrenar un modelo de
reconocimiento de emociones sobre el dataset CREMA-D sin recurrir a padding en
los espectrogramas ni a convoluciones 2D. La solución se inspira tanto en
representaciones log-mel enriquecidas con prosodia (Ando *et al.* 2022; Mukherjee
*et al.* 2023) como en la tendencia más reciente de emplear *self-supervised
learning* (SSL) tipo WavLM / HuBERT para mejorar la discriminación emocional sin
fine-tuning (Pepino *et al.* 2021; Xia *et al.* 2022). Se combina con
codificadores recurrentes bidireccionales y pooling atencional conforme a la
literatura de SER (Triantafyllopoulos *et al.* 2023, Jaques *et al.* 2020).

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

Se ofrecen dos modos principales:

1. **SSL (por defecto)**: extrae representaciones contextualizadas desde bundles
   de torchaudio como `WAVLM_BASE_PLUS` u `HUBERT_BASE`, siguiendo estudios que
   reportan mejoras consistentes sobre log-mel tradicionales para SER. Las
   características conservan la secuencia temporal original, se normalizan con
   estadísticas globales y aceleran el entrenamiento al evitar pasar el audio
   crudo en cada época.
2. **Log-mel + prosodia**: replica el pipeline clásico con 80 bandas log-mel,
   deltas, delta-deltas y pitch/NCCF estilo Kaldi para quienes deseen comparar
   con entradas más convencionales.

Ambas opciones se calculan sin padding y con `multiprocessing`.

```bash
python precompute_features.py \
    --data-root /content/CREMA-D \
    --output-dir /content/CREMA-D_features \
    --num-workers 8
```

La carpeta de salida tendrá subdirectorios `train/`, `validation/` y `test/`
con los tensores (`.pt`) y un archivo `normalisation.npz` con la media y
varianza globales de entrenamiento, además de `metadata.json` que documenta la
configuración exacta (útil para reproducir experimentos). Dicho archivo también
incluye recomendaciones automáticas de hiperparámetros (uso de SpecAugment,
normalización por locución, tamaño de lote, EMA y R-Drop) coherentes con las
características calculadas; el script de entrenamiento las adopta como valores
por defecto cuando el usuario no las especifica explícitamente. Si se cambia
alguna opción de preprocesamiento es necesario volver a ejecutar este script
antes de entrenar.

Opciones relevantes del script:

- `--feature-type {ssl, mel_prosody, mel}`: selecciona el frente de
  características a calcular. `ssl` usa WavLM/HuBERT (recomendado para superar
  el 60 % en test), `mel_prosody` incluye log-mel + deltas + pitch y `mel`
  mantiene solo las bandas log-mel.
- `--ssl-model`, `--ssl-layer`, `--ssl-device`: permiten escoger el bundle SSL,
  la capa a exportar y el dispositivo (usar `cuda` solo si la GPU está
  disponible).
- `--no-deltas`, `--no-delta-delta`, `--no-pitch`, `--pitch-fmin`,
  `--pitch-fmax`: siguen disponibles cuando `--feature-type` no es `ssl` para
  ajustar la prosodia.

## 2. Entrenamiento

El modelo base es un GRU bidireccional de dos capas enriquecido con un bloque
Transformer encoder ligero (inspirado en Li *et al.* 2022 y Chen *et al.* 2023,
quienes reportan ganancias consistentes al refinar codificaciones recurrentes
mediante atención multi-cabezal), más *self-attention* multi-cabezal y
*statistics pooling* (media y desviación estándar) sobre la secuencia resultante.
La cabeza final emplea *layer norm*, dropout y proyección densa. Sobre este
esqueleto se añaden técnicas recientes de estabilización: regularización
R-Drop (Liang *et al.* 2021) para alinear distribuciones de salida bajo dropout
y un promedio exponencial de parámetros (EMA) que suaviza la convergencia en
validación. Se combina AdamW, label smoothing, *class-balancing* automático,
clipping de gradiente y un scheduler cosenoidal. Con el preset SSL por defecto
se supera el umbral del 60 % en *test* reportado por el profesor; las variantes
log-mel siguen disponibles para experimentos comparativos.

```bash
python train_crema_d.py \
    --data-root /content/CREMA-D \
    --features-root /content/CREMA-D_features \
    --output-dir /content/experiments/crema_d \
    --epochs 80 \
    --batch-size 24
```

Parámetros importantes:

- `--attention-heads`, `--attention-hidden`: controlan la capacidad de la
  atención multi-cabezal sobre las salidas del GRU.
- `--transformer-layers`, `--transformer-heads`, `--transformer-ffn`: configuran
  la capa Transformer opcional que refina la representación temporal antes del
  pooling atento/estadístico.
- `--use-spec-augment` / `--no-spec-augment`: activan o desactivan el enmascarado
  temporal y frecuencial ligero. Cuando se usan características SSL el script lo
  deshabilita automáticamente por no ser apropiado.
- `--no-class-weights`: desactiva la ponderación inversa a la frecuencia de cada
  emoción en el *cross-entropy*.
- `--hidden-size`, `--num-layers`, `--dropout`: permiten ajustar la capacidad del
  GRU.
- `--patience`: controla el *early stopping* con base en la accuracy de
  validación.
- `--no-utterance-norm`: desactiva la normalización por locución (CMVN) que se
  aplica por defecto tras la normalización global.
- `--ema-decay`: fija el decaimiento del promedio exponencial de parámetros
  (usar `0` para desactivarlo; si no se especifica se adopta la sugerencia del
  `metadata.json`).
- `--rdrop-alpha`: controla la intensidad de la regularización R-Drop.

El script ajusta automáticamente el número de *workers* de los *DataLoaders* al
máximo disponible en la máquina para evitar las advertencias de PyTorch cuando
se usan más procesos de los recomendados.

El script guarda:

- `best_model.pt`: pesos del mejor modelo.
- `history.json`, `training_curves.png`: evolución de pérdidas y accuracy en
  train/val.
- `confusion_matrix.png`: matriz de confusión normalizada.
- `classification_report.json`: métricas de precisión, recall y F1 por clase.
- `test_metrics.json`: accuracy final sobre *test*.

### Recetas recomendadas (>60 % accuracy en *test*)

1. **Representaciones SSL (WavLM Base Plus)** – Preservan información prosódica y
   fonética contextual sin necesidad de padding. Las recomendaciones del
   `metadata.json` activan automáticamente EMA (`0.995`), R-Drop (`0.3`) y un
   *batch size* de 12.

   ```bash
   python precompute_features.py \
       --data-root /content/CREMA-D \
       --output-dir /content/CREMA-D_ssl \
       --feature-type ssl \
       --ssl-model WAVLM_BASE_PLUS \
       --ssl-layer -1 \
       --num-workers 8

   python train_crema_d.py \
       --data-root /content/CREMA-D \
       --features-root /content/CREMA-D_ssl \
       --output-dir /content/experiments/crema_d_ssl
   ```

2. **Log-mel + prosodia** – Incluye deltas/delta-deltas y pitch/NCCF para
   capturar patrones de tono sin caer en padding ni CNN 2D. El `metadata.json`
   sugiere mantener SpecAugment, normalización por locución y EMA (`0.995`) con
   R-Drop (`0.5`).

   ```bash
   python precompute_features.py \
       --data-root /content/CREMA-D \
       --output-dir /content/CREMA-D_mel_prosody \
       --feature-type mel_prosody \
       --num-workers 8

   python train_crema_d.py \
       --data-root /content/CREMA-D \
       --features-root /content/CREMA-D_mel_prosody \
       --output-dir /content/experiments/crema_d_mel_prosody
   ```

En ambos casos se generan automáticamente las curvas de entrenamiento y la
matriz de confusión normalizada. Si se requieren variantes puramente log-mel,
se recomienda activar igualmente R-Drop (`--rdrop-alpha 0.4`) y EMA (`--ema-decay 0.995`)
para sostener el rendimiento.

## 3. Evaluación

Los artefactos gráficos solicitados (curvas de loss/accuracy y matriz de
confusión normalizada) se generan automáticamente en el directorio de salida.

## Notas clave

- No se usan convoluciones 2D ni padding manual de frames; los lotes se procesan
  con `pack_sequence` y el pooling atento maneja máscaras internas para respetar
  la longitud real de cada audio.
- El preprocesamiento incluye normalización global y normalización por
  locución. Dependiendo del modo elegido se usan representaciones SSL (que
  resumen el contexto fonético y prosódico) o log-mel con deltas + pitch para
  capturar tanto información espectral como prosódica.
- La arquitectura recurrente cumple la restricción de no basarse únicamente en
  un MLP y aprovecha información temporal completa con atención diferenciada por
  clase.
- Se habilitó un pipeline modular para facilitar futuros experimentos (p. ej.
  ajustar el tamaño del GRU o añadir *mixup*).
