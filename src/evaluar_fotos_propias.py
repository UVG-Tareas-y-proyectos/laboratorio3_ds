"""Evalua el mejor modelo con las fotos de Ihan y Diego.

Estructura esperada (minimo cinco letras distintas por integrante):

    data/propias/Ihan/A_01.jpg
    data/propias/Ihan/B_01.jpg
    data/propias/Diego/A_01.jpg

Tambien se acepta ``data/propias/Ihan/A/01.jpg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from modelos import (
    cargar_checkpoint,
    evaluar_fotos_propias,
    evaluar_fotos_random_forest,
)


BASE = Path(__file__).resolve().parent.parent


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE / "resultados" / "mejor_modelo.pt",
    )
    parser.add_argument(
        "--random-forest",
        type=Path,
        default=BASE / "resultados" / "mejor_modelo_rf.joblib",
    )
    parser.add_argument(
        "--modelo",
        choices=("auto", "rf", "cnn"),
        default="auto",
        help="auto usa Random Forest si existe; cnn permite comparar la mejor red.",
    )
    parser.add_argument(
        "--fotos", type=Path, default=BASE / "data" / "propias"
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=BASE / "resultados" / "fotos_propias.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    tipo_modelo = (
        "rf"
        if args.modelo == "auto" and args.random_forest.exists()
        else args.modelo
    )
    if tipo_modelo == "auto":
        tipo_modelo = "cnn"

    if tipo_modelo == "rf":
        modelo = joblib.load(args.random_forest)
        clases = np.load(BASE / "data" / "processed" / "clases.npy")
        resultados = evaluar_fotos_random_forest(modelo, clases, args.fotos)
        nombre_modelo = "Random Forest"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo, clases = cargar_checkpoint(args.checkpoint, device)
        resultados = evaluar_fotos_propias(modelo, clases, args.fotos, device)
        nombre_modelo = "CNN"
    if resultados.empty:
        raise FileNotFoundError(
            f"No se encontraron fotos en {args.fotos}. "
            "Agregue al menos cinco letras distintas por integrante."
        )

    conteos = resultados.groupby("integrante")["real"].nunique()
    incompletos = conteos[conteos < 5]
    if not incompletos.empty:
        detalle = ", ".join(f"{persona}: {n}" for persona, n in incompletos.items())
        raise ValueError(
            "Cada integrante necesita al menos cinco letras distintas. "
            f"Actualmente: {detalle}."
        )

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(args.salida, index=False)
    resumen = (
        resultados.groupby("integrante")
        .agg(
            fotos=("acierto", "size"),
            letras=("real", "nunique"),
            accuracy=("acierto", "mean"),
        )
        .reset_index()
    )
    print("Modelo:", nombre_modelo)
    print(resultados[["integrante", "real", "predicha", "confianza", "acierto"]])
    print("\nResumen por integrante:\n", resumen.to_string(index=False))
    print("\nResultados guardados en:", args.salida)


if __name__ == "__main__":
    main()
