"""Prepara ASL Alphabet con una submuestra estratificada y reproducible.

Acepta el dataset extraido o el ZIP descargado de Kaggle. Genera arreglos ``.npy``
uint8 para train/validacion/prueba y un ``metadata.json`` con la trazabilidad del
preprocesamiento. La normalizacion a [0, 1] se hace de forma perezosa al entrenar
para no triplicar el consumo de memoria.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DIR_PREDETERMINADO = RAW / "asl_alphabet_train"
ZIP_PREDETERMINADO = RAW / "asl_alphabet_train.zip"
OUT_PREDETERMINADO = BASE / "data" / "processed"

TAM = 64
POR_CLASE = 500
SEED = 0
EXTENSIONES = {".jpg", ".jpeg", ".png"}
CLASES_ESPERADAS = {
    *(chr(codigo) for codigo in range(ord("A"), ord("Z") + 1)),
    "del",
    "nothing",
    "space",
}


def _es_imagen(nombre: str | Path) -> bool:
    return Path(nombre).suffix.lower() in EXTENSIONES


def descubrir_en_directorio(directorio: Path) -> dict[str, list[Path]]:
    """Agrupa imagenes por el nombre de su carpeta hoja.

    ``rglob`` permite leer tanto ``.../asl_alphabet_train/A`` como la estructura
    duplicada ``.../asl_alphabet_train/asl_alphabet_train/A`` de algunas
    extracciones del ZIP de Kaggle.
    """

    por_clase: dict[str, list[Path]] = defaultdict(list)
    for ruta in directorio.rglob("*"):
        if ruta.is_file() and _es_imagen(ruta):
            por_clase[ruta.parent.name].append(ruta)
    return {clase: sorted(rutas) for clase, rutas in sorted(por_clase.items())}


def descubrir_en_zip(ruta_zip: Path) -> tuple[zipfile.ZipFile, dict[str, list[str]]]:
    zf = zipfile.ZipFile(ruta_zip)
    por_clase: dict[str, list[str]] = defaultdict(list)
    for info in zf.infolist():
        if not info.is_dir() and _es_imagen(info.filename):
            partes = Path(info.filename).parts
            if len(partes) >= 2:
                por_clase[partes[-2]].append(info.filename)
    return zf, {clase: sorted(rutas) for clase, rutas in sorted(por_clase.items())}


def resolver_fuente(directorio: Path, ruta_zip: Path) -> tuple[str, object, dict[str, list]]:
    if directorio.is_dir():
        por_clase = descubrir_en_directorio(directorio)
        if por_clase:
            return "directorio", directorio, por_clase

    candidatos = [ruta_zip]
    if ruta_zip == ZIP_PREDETERMINADO and RAW.exists():
        candidatos.extend(sorted(RAW.glob("*.zip")))
    for candidato in dict.fromkeys(candidatos):
        if candidato.exists():
            zf, por_clase = descubrir_en_zip(candidato)
            if por_clase:
                return "zip", zf, por_clase
            zf.close()

    raise FileNotFoundError(
        "No se encontro ASL Alphabet. Coloque la carpeta extraida en "
        f"{directorio} o un ZIP de Kaggle en {RAW}."
    )


def cargar_imagen(fuente: Path | tuple[zipfile.ZipFile, str], tam: int) -> np.ndarray:
    if isinstance(fuente, tuple):
        zf, nombre = fuente
        archivo = io.BytesIO(zf.read(nombre))
    else:
        archivo = fuente
    with Image.open(archivo) as imagen:
        imagen = imagen.convert("RGB").resize((tam, tam), Image.Resampling.BILINEAR)
        return np.asarray(imagen, dtype=np.uint8)


def indices_estratificados(
    y: np.ndarray, n_clases: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye un split 70/15/15 sin mezclar el conjunto de prueba en el tuneo."""

    train, validacion, prueba = [], [], []
    for clase in range(n_clases):
        indices = np.flatnonzero(y == clase)
        rng.shuffle(indices)
        n = len(indices)
        corte_train = int(n * 0.70)
        corte_val = int(n * 0.85)
        train.extend(indices[:corte_train])
        validacion.extend(indices[corte_train:corte_val])
        prueba.extend(indices[corte_val:])

    salida = []
    for indices in (train, validacion, prueba):
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        salida.append(indices)
    return tuple(salida)  # type: ignore[return-value]


def preparar(
    directorio: Path = DIR_PREDETERMINADO,
    ruta_zip: Path = ZIP_PREDETERMINADO,
    salida: Path = OUT_PREDETERMINADO,
    tam: int = TAM,
    por_clase_objetivo: int = POR_CLASE,
    seed: int = SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    tipo_fuente, fuente_abierta, por_clase = resolver_fuente(directorio, ruta_zip)

    clases_encontradas = set(por_clase)
    faltantes = sorted(CLASES_ESPERADAS - clases_encontradas)
    if faltantes:
        if tipo_fuente == "zip":
            fuente_abierta.close()
        raise ValueError(
            "No se encontraron todas las clases de entrenamiento de ASL Alphabet. "
            f"Faltan: {faltantes}"
        )

    # El ZIP oficial tambien contiene ``asl_alphabet_test/asl_alphabet_test``.
    # Esa carpeta no es una clase y se excluye para evitar fuga de datos.
    clases_ignoradas = sorted(clases_encontradas - CLASES_ESPERADAS)
    por_clase = {clase: por_clase[clase] for clase in sorted(CLASES_ESPERADAS)}
    clases = list(por_clase)

    imagenes: list[np.ndarray] = []
    etiquetas: list[int] = []
    conteos_originales = {clase: len(por_clase[clase]) for clase in clases}

    for etiqueta, clase in enumerate(clases):
        rutas = por_clase[clase]
        cantidad = min(por_clase_objetivo, len(rutas))
        seleccion = rng.choice(len(rutas), size=cantidad, replace=False)
        for indice in seleccion:
            ruta = rutas[int(indice)]
            origen = (fuente_abierta, ruta) if tipo_fuente == "zip" else ruta
            imagenes.append(cargar_imagen(origen, tam))
            etiquetas.append(etiqueta)

    if tipo_fuente == "zip":
        nombre_fuente = str(fuente_abierta.filename)
        fuente_abierta.close()
    else:
        nombre_fuente = str(fuente_abierta)

    X = np.stack(imagenes)
    y = np.asarray(etiquetas, dtype=np.int64)
    idx_train, idx_val, idx_test = indices_estratificados(y, len(clases), rng)

    salida.mkdir(parents=True, exist_ok=True)
    arreglos = {
        "train": (X[idx_train], y[idx_train]),
        "val": (X[idx_val], y[idx_val]),
        "test": (X[idx_test], y[idx_test]),
    }
    for split, (imagenes_split, etiquetas_split) in arreglos.items():
        np.save(salida / f"X_{split}.npy", imagenes_split)
        np.save(salida / f"y_{split}.npy", etiquetas_split)
    np.save(salida / "clases.npy", np.asarray(clases))

    metadata = {
        "dataset": "ASL Alphabet (grassknoted/asl-alphabet)",
        "fuente": nombre_fuente,
        "formato_original": "JPG RGB",
        "resolucion_original": [200, 200, 3],
        "resolucion_procesada": [tam, tam, 3],
        "dtype_almacenado": "uint8",
        "normalizacion_entrenamiento": "x / 255.0",
        "seed": seed,
        "clases": clases,
        "clases_ignoradas": clases_ignoradas,
        "conteos_originales": conteos_originales,
        "submuestra_por_clase": {
            clase: int(np.sum(y == i)) for i, clase in enumerate(clases)
        },
        "split": {nombre: int(len(datos[1])) for nombre, datos in arreglos.items()},
        "proporciones": {"train": 0.70, "val": 0.15, "test": 0.15},
    }
    (salida / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("clases:", len(clases))
    print("train/val/test:", *(len(arreglos[s][1]) for s in ("train", "val", "test")))
    print("shape imagen:", X.shape[1:])
    print("metadata:", salida / "metadata.json")
    return metadata


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directorio", type=Path, default=DIR_PREDETERMINADO)
    parser.add_argument("--zip", dest="ruta_zip", type=Path, default=ZIP_PREDETERMINADO)
    parser.add_argument("--salida", type=Path, default=OUT_PREDETERMINADO)
    parser.add_argument("--tam", type=int, default=TAM)
    parser.add_argument("--por-clase", type=int, default=POR_CLASE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parsear_argumentos()
    preparar(
        directorio=args.directorio,
        ruta_zip=args.ruta_zip,
        salida=args.salida,
        tam=args.tam,
        por_clase_objetivo=args.por_clase,
        seed=args.seed,
    )
