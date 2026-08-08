# Laboratorio 3 - Deep Learning (reconocimiento de señas ASL)

Solución reproducible del laboratorio de CC3084 Data Science sobre el dataset
[ASL Alphabet de Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
Incluye EDA, preprocesamiento, cuatro configuraciones CNN, una red fully-connected,
Random Forest, image augmentation, comparación de modelos y evaluación de fotos
propias.

## Preparación

1. Cree un ambiente e instale las dependencias:

   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   python -m pip install -r requirements.txt
   ```

2. Descargue el dataset y coloque uno de estos insumos (no se versionan por tamaño):

   - carpeta extraída: `data/raw/asl_alphabet_train/`; o
   - ZIP de Kaggle: `data/raw/asl_alphabet_train.zip` (también se detecta cualquier
     ZIP único dentro de `data/raw/`).

3. Prepare la submuestra balanceada de 500 imágenes por clase, a 64x64 RGB y con
   split estratificado 70/15/15:

   ```bash
   python src/preparar_datos.py
   ```

4. Ejecute `notebooks/laboratorio3.ipynb` de inicio a fin. El conjunto de prueba
   se usa una sola vez después de elegir hiperparámetros con validación.

## Fotos propias

Agregue al menos cinco letras distintas por integrante. Se aceptan ambos formatos:

```text
data/propias/Ihan/A_01.jpg
data/propias/Ihan/B_01.jpg
data/propias/Diego/A/01.jpg
data/propias/Diego/B/01.jpg
```

La última parte del notebook guarda `resultados/mejor_modelo.pt` y evalúa las fotos.
También puede repetir solamente esa evaluación con:

```bash
python src/evaluar_fotos_propias.py
```

El script verifica que cada persona tenga por lo menos cinco etiquetas distintas y
genera `resultados/fotos_propias.csv`.

## Estructura

- `notebooks/laboratorio3.ipynb`: análisis completo y discusión de resultados.
- `src/preparar_datos.py`: lectura robusta, submuestreo, redimensionado y splits.
- `src/modelos.py`: CNN, red densa, Random Forest, aumentos, métricas y checkpoints.
- `src/evaluar_fotos_propias.py`: evaluación repetible de imágenes del grupo.
- `data/`: dataset y arreglos procesados; ignorados por Git.
- `resultados/`: checkpoints y CSV; ignorados por Git.

## Decisiones metodológicas

- Se conserva RGB y se normaliza a `[0, 1]` durante el entrenamiento.
- No se aplica flip horizontal: en una seña asimétrica puede alterar orientación,
  lateralidad o significado, no solo crear una vista equivalente.
- Los aumentos válidos son rotación leve, traslación, escala, cambios moderados de
  iluminación/contraste y desenfoque ocasional.
- Random Forest usa imágenes grises reducidas a 16x16 para mantener un baseline
  clásico interpretable y computacionalmente viable.
- Las métricas principales son accuracy y F1 macro porque las 29 clases quedan
  balanceadas después del submuestreo.
