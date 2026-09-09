"""Funciones auxiliares de la clase 03.

Versión reducida: solo lo que la letra importa (get_device, get_num_workers, plot_training).
Las funciones de entrenamiento, early stopping y reporte de clasificación son los ejercicios de
esta clase; a partir de la clase 04 viven en utils.py ya resueltas.
"""

import os
import sys

import torch
import matplotlib.pyplot as plt


def get_device():
    """
    Devuelve el mejor dispositivo disponible para PyTorch, en orden de preferencia:
    "cuda" (GPU NVIDIA), "mps" (GPU Apple Silicon), "xpu" (GPU Intel) o "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # hasattr: versiones viejas de torch no tienen torch.xpu
        return "xpu"
    return "cpu"


def get_num_workers(fraction=0.5, min_workers=1, max_workers=8):
    """
    Devuelve la cantidad de procesos (num_workers) para los DataLoaders según el sistema operativo y los núcleos disponibles.

    num_workers > 0 hace que el DataLoader cargue los batches en procesos hijos, en paralelo.
    En Linux los hijos se crean con fork (copia del proceso actual, con el Dataset ya en memoria).
    En Windows y macOS se crean con spawn: un intérprete nuevo que debe reconstruir el Dataset
    importándolo por nombre, y las clases definidas en un notebook no son importables -> errores de pickle.
    Por eso en Windows y macOS devolvemos siempre 0 (los datos se cargan en el proceso principal).

    En Linux usamos una fracción de los núcleos disponibles, acotada entre min_workers y max_workers:
    más workers que núcleos no acelera nada (compiten por la CPU con el proceso principal) y cada
    worker consume memoria. Los núcleos se cuentan con os.sched_getaffinity, que respeta los límites
    del contenedor o de la máquina virtual (Colab, Docker), a diferencia de os.cpu_count().

    Args:
        fraction (float): Fracción de los núcleos disponibles a usar (default: 0.5).
        min_workers (int): Mínimo de workers (default: 1).
        max_workers (int): Máximo de workers (default: 8). Importa sobre todo cuando cada muestra se lee
            de disco o pasa por transformaciones (imágenes); con datos ya en memoria, pocos workers alcanzan.
            Para fijar un valor exacto, usar min_workers = max_workers.

    Returns:
        int: 0 en Windows/macOS; en Linux, floor(núcleos * fraction) acotado a [min_workers, max_workers].
    """
    if sys.platform != "linux":
        return 0
    try:
        cpus = len(os.sched_getaffinity(0))  # núcleos que este proceso puede usar
    except AttributeError:  # no disponible en algunas plataformas
        cpus = os.cpu_count() or 1
    return max(min_workers, min(max_workers, int(cpus * fraction)))


def plot_training(train_errors, val_errors, mark_min=False, title="Training and Validation Loss"):
    """
    Grafica la pérdida de entrenamiento y validación por época.

    Args:
        train_errors (list[float]): Pérdida de entrenamiento por época.
        val_errors (list[float]): Pérdida de validación por época.
        mark_min (bool, optional): Si es True, marca con una línea vertical la época con menor pérdida de validación.
        title (str, optional): Título del gráfico.
    """
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    if mark_min:
        best_epoch = min(range(len(val_errors)), key=lambda i: val_errors[i])
        plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Mín. val loss (época {best_epoch + 1})")
    plt.title(title)  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico
