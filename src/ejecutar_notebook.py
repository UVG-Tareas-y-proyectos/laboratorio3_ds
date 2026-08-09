"""Ejecuta el notebook completo en segundo plano y registra su estado.

El notebook original solo se reemplaza cuando todas las celdas terminan sin
errores. Mientras se ejecuta, una copia incremental queda en ``tmp`` y el
estado puede consultarse en ``resultados/ejecucion_estado.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient


BASE = Path(__file__).resolve().parent.parent
NOTEBOOK = BASE / "notebooks" / "laboratorio3.ipynb"
COPIA_TRABAJO = BASE / "tmp" / "laboratorio3_ejecutando.ipynb"
ESTADO = BASE / "resultados" / "ejecucion_estado.json"


def ahora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def guardar_estado(**datos: object) -> None:
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    temporal = ESTADO.with_suffix(".json.tmp")
    temporal.write_text(
        json.dumps({"pid": os.getpid(), "actualizado": ahora(), **datos}, indent=2),
        encoding="utf-8",
    )
    temporal.replace(ESTADO)


def texto_salida(celda: nbformat.NotebookNode) -> str:
    partes: list[str] = []
    for salida in celda.get("outputs", []):
        if salida.output_type == "stream":
            partes.append(salida.get("text", ""))
        elif salida.output_type in {"execute_result", "display_data"}:
            texto = salida.get("data", {}).get("text/plain")
            if texto:
                partes.append(texto)
    return "\n".join(partes).strip()


def main() -> None:
    inicio = ahora()
    COPIA_TRABAJO.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for celda in notebook.cells:
        if celda.cell_type == "code":
            celda.outputs = []
            celda.execution_count = None
    nbformat.write(notebook, COPIA_TRABAJO)

    cliente = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        allow_errors=False,
    )
    guardar_estado(estado="iniciando", inicio=inicio, total_celdas=len(notebook.cells))

    try:
        ejecutadas = 0
        with cliente.setup_kernel():
            for indice, celda in enumerate(notebook.cells):
                if celda.cell_type != "code":
                    continue
                primera_linea = next(
                    (linea.strip() for linea in celda.source.splitlines() if linea.strip()),
                    "celda vacia",
                )
                guardar_estado(
                    estado="ejecutando",
                    inicio=inicio,
                    celda=indice,
                    descripcion=primera_linea[:160],
                    celdas_codigo_completadas=ejecutadas,
                )
                print(f"[{ahora()}] celda {indice}: {primera_linea}", flush=True)
                cliente.execute_cell(celda, indice)
                ejecutadas += 1
                nbformat.write(notebook, COPIA_TRABAJO)
                salida = texto_salida(celda)
                if salida:
                    print(salida, flush=True)

        shutil.copy2(COPIA_TRABAJO, NOTEBOOK)
        guardar_estado(
            estado="completado",
            inicio=inicio,
            fin=ahora(),
            celdas_codigo_completadas=ejecutadas,
            notebook=str(NOTEBOOK),
        )
        print(f"[{ahora()}] notebook completado", flush=True)
    except Exception as exc:
        nbformat.write(notebook, COPIA_TRABAJO)
        guardar_estado(
            estado="error",
            inicio=inicio,
            fin=ahora(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            copia_parcial=str(COPIA_TRABAJO),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
