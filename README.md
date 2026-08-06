# Laboratorio 3 - Deep Learning (Reconocimiento de señas ASL)

Avance: análisis exploratorio, preprocesamiento y modelos CNN con tuneo de parámetros sobre el
dataset ASL Alphabet (29 clases).

## Correr

El zip del dataset (`asl_alphabet_train.zip`) va en `data/raw/` (no se versiona por tamaño).

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/jupyter notebook notebooks/laboratorio3.ipynb
```

## Estructura

- `notebooks/laboratorio3.ipynb` - EDA, preprocesamiento, CNN y tuneo, elección del mejor modelo
- `src/preparar_datos.py` - lee el zip, submuestrea, redimensiona y arma el split (genera `data/processed/`)
- `data/raw/` - zip del dataset, no se modifica ni se versiona
- `data/processed/` - arrays `.npy` de train/val/test
