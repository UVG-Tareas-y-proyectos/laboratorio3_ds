"""Prepara el dataset ASL: submuestrea, redimensiona y arma los splits.
Genera los .npy de data/processed/. Lee de la carpeta extraida en data/raw/ (o del zip si aun existe)."""
import io
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DIR = BASE / "data/raw/asl_alphabet_train"      # carpeta extraida
ZIP = BASE / "data/raw/asl_alphabet_train.zip"  # o el zip original
OUT = BASE / "data/processed"

TAM = 64        # imagenes a 64x64
POR_CLASE = 500  # submuestra por clase (el dataset tiene 3000, con esto entrena en tiempo razonable)
SEED = 0

rng = np.random.default_rng(SEED)


def cargar_imagen(fuente):
    """fuente: ruta en disco o (zip, nombre)."""
    if isinstance(fuente, tuple):
        z, nombre = fuente
        data = io.BytesIO(z.read(nombre))
    else:
        data = fuente
    return np.asarray(Image.open(data).convert("RGB").resize((TAM, TAM)), dtype=np.uint8)


# agrupar rutas de imagenes por clase (carpeta), desde la carpeta o el zip
zf = None
if DIR.is_dir():
    por_clase = {p.name: [f for f in p.glob("*.jpg")] for p in sorted(DIR.iterdir()) if p.is_dir()}
elif ZIP.exists():
    zf = zipfile.ZipFile(ZIP)
    por_clase = {}
    for info in zf.infolist():
        if info.filename.endswith(".jpg"):
            por_clase.setdefault(info.filename.split("/")[-2], []).append((zf, info.filename))
else:
    raise FileNotFoundError(f"No se encontro el dataset en {DIR} ni {ZIP}")

clases = sorted(por_clase)
clase_a_idx = {c: i for i, c in enumerate(clases)}

X, y = [], []
for c in clases:
    rutas = por_clase[c]
    elegidas = rng.choice(len(rutas), size=min(POR_CLASE, len(rutas)), replace=False)
    for i in elegidas:
        X.append(cargar_imagen(rutas[i]))
        y.append(clase_a_idx[c])

X = np.stack(X)
y = np.array(y, dtype=np.int64)

# split estratificado 70/15/15 barajando cada clase por separado
tr_i, va_i, te_i = [], [], []
for c in range(len(clases)):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n = len(idx)
    a, b = int(n * 0.70), int(n * 0.85)
    tr_i += list(idx[:a]); va_i += list(idx[a:b]); te_i += list(idx[b:])
tr_i, va_i, te_i = map(np.array, (tr_i, va_i, te_i))

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "X_train.npy", X[tr_i]); np.save(OUT / "y_train.npy", y[tr_i])
np.save(OUT / "X_val.npy", X[va_i]);   np.save(OUT / "y_val.npy", y[va_i])
np.save(OUT / "X_test.npy", X[te_i]);  np.save(OUT / "y_test.npy", y[te_i])
np.save(OUT / "clases.npy", np.array(clases))

print("clases:", len(clases))
print("train/val/test:", len(tr_i), len(va_i), len(te_i))
print("shape imagen:", X.shape[1:])
