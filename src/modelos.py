"""Modelos y utilidades reproducibles para el laboratorio de ASL Alphabet."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


def fijar_semilla(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetImagenes(Dataset):
    """Convierte uint8 NHWC a tensores CHW sin duplicar todo el dataset en RAM."""

    def __init__(
        self,
        imagenes: np.ndarray,
        etiquetas: np.ndarray,
        transformacion: Callable | None = None,
    ) -> None:
        self.imagenes = imagenes
        self.etiquetas = etiquetas
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        imagen = self.imagenes[indice]
        if self.transformacion is None:
            tensor = torch.from_numpy(imagen.copy()).permute(2, 0, 1).float().div(255.0)
        else:
            tensor = self.transformacion(imagen)
        etiqueta = torch.tensor(int(self.etiquetas[indice]), dtype=torch.long)
        return tensor, etiqueta


def transformacion_aumento():
    """Aumentos plausibles para senas; excluye flips y rotaciones extremas."""

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=(-5, 5),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.10, 1.00))], p=0.15
            ),
            transforms.ToTensor(),
        ]
    )


def crear_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    barajar: bool = False,
    aumentar: bool = False,
) -> DataLoader:
    dataset = DatasetImagenes(X, y, transformacion_aumento() if aumentar else None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=barajar,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class CNN(nn.Module):
    def __init__(
        self,
        n_clases: int,
        bloques: int = 3,
        base: int = 32,
        dropout: float = 0.30,
        tam: int = 64,
    ) -> None:
        super().__init__()
        capas: list[nn.Module] = []
        canales = 3
        for bloque in range(bloques):
            salida = base * (2**bloque)
            capas.extend(
                [
                    nn.Conv2d(canales, salida, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            canales = salida
        lado = tam // (2**bloques)
        self.conv = nn.Sequential(*capas)
        self.cabeza = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(canales * lado * lado, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.conv(x))


class RedDensa(nn.Module):
    """Baseline fully-connected que ignora explicitamente la estructura espacial."""

    def __init__(
        self,
        n_clases: int,
        ocultas: tuple[int, int] = (512, 256),
        dropout: float = 0.40,
        tam: int = 64,
    ) -> None:
        super().__init__()
        entrada = 3 * tam * tam
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(entrada, ocultas[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ocultas[0], ocultas[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ocultas[1], n_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


@dataclass
class ResultadoEntrenamiento:
    modelo: nn.Module
    historial: pd.DataFrame
    mejor_val_acc: float
    mejor_epoca: int


def _paso_epoca(
    modelo: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizador: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida_total = 0.0
    correctas = 0
    total = 0

    contexto = torch.enable_grad() if entrenando else torch.no_grad()
    with contexto:
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if entrenando:
                optimizador.zero_grad(set_to_none=True)
            logits = modelo(xb)
            perdida = loss_fn(logits, yb)
            if entrenando:
                perdida.backward()
                optimizador.step()
            perdida_total += perdida.item() * len(yb)
            correctas += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return perdida_total / total, correctas / total


def entrenar_modelo(
    modelo: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-3,
    paciencia: int = 3,
    seed: int = 0,
) -> ResultadoEntrenamiento:
    fijar_semilla(seed)
    modelo = modelo.to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    filas = []
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_val = -np.inf
    mejor_epoca = 0
    sin_mejora = 0

    for epoca in range(1, epochs + 1):
        train_loss, train_acc = _paso_epoca(
            modelo, dl_train, device, loss_fn, optimizador
        )
        val_loss, val_acc = _paso_epoca(modelo, dl_val, device, loss_fn)
        filas.append(
            {
                "epoca": epoca,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoca {epoca:02d} | train_acc={train_acc:.4f} "
            f"| val_acc={val_acc:.4f}"
        )
        if val_acc > mejor_val:
            mejor_val = val_acc
            mejor_epoca = epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print("early stopping")
                break

    modelo.load_state_dict(mejor_estado)
    return ResultadoEntrenamiento(
        modelo=modelo,
        historial=pd.DataFrame(filas),
        mejor_val_acc=float(mejor_val),
        mejor_epoca=mejor_epoca,
    )


def predecir_pytorch(
    modelo: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modelo.eval()
    reales, predicciones, confianzas = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            probabilidades = torch.softmax(modelo(xb.to(device)), dim=1).cpu()
            confianza, prediccion = probabilidades.max(dim=1)
            reales.append(yb.numpy())
            predicciones.append(prediccion.numpy())
            confianzas.append(confianza.numpy())
    return (
        np.concatenate(reales),
        np.concatenate(predicciones),
        np.concatenate(confianzas),
    )


def metricas_clasificacion(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def features_clasicas(X: np.ndarray, paso: int = 4) -> np.ndarray:
    """Reduce 64x64 RGB a 16x16 gris para un baseline Random Forest manejable."""

    gris = (
        0.299 * X[..., 0].astype(np.float32)
        + 0.587 * X[..., 1].astype(np.float32)
        + 0.114 * X[..., 2].astype(np.float32)
    )
    reducido = gris[:, ::paso, ::paso] / 255.0
    return reducido.reshape(len(X), -1)


def ajustar_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    configuraciones: Iterable[dict],
    seed: int = 0,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict]:
    filas = []
    mejor_modelo = None
    mejor_cfg = None
    mejor_acc = -np.inf
    for cfg in configuraciones:
        modelo = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            **cfg,
        )
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_val, modelo.predict(X_val))
        filas.append({**cfg, "val_acc": float(acc)})
        print(cfg, "-> val_acc:", round(acc, 4))
        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo
            mejor_cfg = dict(cfg)
    assert mejor_modelo is not None and mejor_cfg is not None
    return mejor_modelo, pd.DataFrame(filas).sort_values("val_acc", ascending=False), mejor_cfg


def cargar_foto(ruta: Path, tam: int = 64) -> torch.Tensor:
    with Image.open(ruta) as imagen:
        arreglo = np.asarray(
            imagen.convert("RGB").resize((tam, tam), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return torch.from_numpy(arreglo.copy()).permute(2, 0, 1).float().div(255.0)


def descubrir_fotos_propias(directorio: Path) -> list[tuple[str, str, Path]]:
    """Lee ``persona/clase/*.jpg`` o ``persona/A_01.jpg``."""

    filas = []
    if not directorio.exists():
        return filas
    for persona_dir in sorted(p for p in directorio.iterdir() if p.is_dir()):
        for ruta in sorted(persona_dir.rglob("*")):
            if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if ruta.parent != persona_dir:
                etiqueta = ruta.parent.name
            else:
                etiqueta = ruta.stem.split("_")[0].split("-")[0]
            filas.append((persona_dir.name, etiqueta, ruta))
    return filas


def evaluar_fotos_propias(
    modelo: nn.Module,
    clases: np.ndarray,
    directorio: Path,
    device: torch.device,
) -> pd.DataFrame:
    filas = descubrir_fotos_propias(directorio)
    if not filas:
        return pd.DataFrame(
            columns=["integrante", "archivo", "real", "predicha", "confianza", "acierto"]
        )
    modelo.eval()
    resultados = []
    with torch.no_grad():
        for integrante, real, ruta in filas:
            tensor = cargar_foto(ruta).unsqueeze(0).to(device)
            probabilidades = torch.softmax(modelo(tensor), dim=1)[0]
            confianza, indice = probabilidades.max(dim=0)
            predicha = str(clases[int(indice)])
            resultados.append(
                {
                    "integrante": integrante,
                    "archivo": str(ruta),
                    "real": real,
                    "predicha": predicha,
                    "confianza": float(confianza),
                    "acierto": real.lower() == predicha.lower(),
                }
            )
    return pd.DataFrame(resultados)


def guardar_checkpoint(
    ruta: Path,
    modelo: nn.Module,
    arquitectura: str,
    configuracion: dict,
    clases: np.ndarray,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arquitectura": arquitectura,
            "configuracion": configuracion,
            "clases": [str(x) for x in clases],
            "state_dict": modelo.state_dict(),
        },
        ruta,
    )


def cargar_checkpoint(ruta: Path, device: torch.device) -> tuple[nn.Module, np.ndarray]:
    datos = torch.load(ruta, map_location=device)
    clases = np.asarray(datos["clases"])
    configuracion = dict(datos["configuracion"])
    if datos["arquitectura"] == "CNN":
        modelo = CNN(len(clases), **configuracion)
    elif datos["arquitectura"] == "RedDensa":
        if isinstance(configuracion.get("ocultas"), list):
            configuracion["ocultas"] = tuple(configuracion["ocultas"])
        modelo = RedDensa(len(clases), **configuracion)
    else:
        raise ValueError(f"Arquitectura no soportada: {datos['arquitectura']}")
    modelo.load_state_dict(datos["state_dict"])
    return modelo.to(device).eval(), clases
