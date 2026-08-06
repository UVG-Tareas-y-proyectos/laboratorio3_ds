"""Prepara el dataset ASL: lee el zip, submuestrea, redimensiona y arma los splits.
Genera los .npy de data/processed/. Se lee directo del zip para no extraer 87k archivos."""
import io
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
ZIP = BASE / "data/raw/asl_alphabet_train.zip"
OUT = BASE / "data/processed"

TAM = 64        # imagenes a 64x64
POR_CLASE = 500  # submuestra por clase (el dataset tiene 3000, con esto entrena en tiempo razonable)
SEED = 0

rng = np.random.default_rng(SEED)

with zipfile.ZipFile(ZIP) as z:
    # agrupar rutas de imagenes por clase (carpeta)
    por_clase = {}
    for info in z.infolist():
        if info.filename.endswith(".jpg"):
            clase = info.filename.split("/")[-2]
            por_clase.setdefault(clase, []).append(info.filename)

    clases = sorted(por_clase)
    clase_a_idx = {c: i for i, c in enumerate(clases)}

    X, y = [], []
    for c in clases:
        rutas = por_clase[c]
        elegidas = rng.choice(rutas, size=min(POR_CLASE, len(rutas)), replace=False)
        for r in elegidas:
            img = Image.open(io.BytesIO(z.read(r))).convert("RGB").resize((TAM, TAM))
            X.append(np.asarray(img, dtype=np.uint8))
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
