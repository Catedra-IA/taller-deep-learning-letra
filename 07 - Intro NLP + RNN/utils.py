import os
import sys

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)


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


def evaluate(model, criterion, data_loader, device):
    """
    Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

    Args:
        model (torch.nn.Module): El modelo que se va a evaluar.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        data_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de evaluación.

    Returns:
        float: La pérdida promedio en el conjunto de datos de evaluación.

    """
    model.eval()  # ponemos el modelo en modo de evaluacion
    total_loss = 0  # acumulador de la perdida
    with torch.no_grad():  # deshabilitamos el calculo de gradientes
        for x, y in data_loader:  # iteramos sobre el dataloader
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo
            output = model(x)  # forward pass
            total_loss += criterion(output, y).item()  # acumulamos la perdida
    return total_loss / len(data_loader)  # retornamos la perdida promedio


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
    )

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    do_early_stopping=True,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de validación.
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        patience (int): Número de épocas a esperar después de la última mejora en val_loss antes de detener el entrenamiento (default: 5).
        epochs (int): Número de épocas de entrenamiento (default: 10).
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss, val_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).

    Returns:
        Tuple[List[float], List[float]]: Una tupla con dos listas, la primera con el error de entrenamiento de cada época y la segunda con el error de validación de cada época.

    """
    epoch_train_errors = []  # colectamos el error de traing para posterior analisis
    epoch_val_errors = []  # colectamos el error de validacion para posterior analisis
    if do_early_stopping:
        early_stopping = EarlyStopping(
            patience=patience
        )  # instanciamos el early stopping

    for epoch in range(epochs):  # loop de entrenamiento
        model.train()  # ponemos el modelo en modo de entrenamiento
        train_loss = 0  # acumulador de la perdida de entrenamiento
        for x, y in train_loader:
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo

            optimizer.zero_grad()  # reseteamos los gradientes

            output = model(x)  # forward pass (prediccion)
            batch_loss = criterion(
                output, y
            )  # calculamos la perdida con la salida esperada

            batch_loss.backward()  # backpropagation
            optimizer.step()  # actualizamos los pesos

            train_loss += batch_loss.item()  # acumulamos la perdida

        train_loss /= len(train_loader)  # calculamos la perdida promedio de la epoca
        epoch_train_errors.append(train_loss)  # guardamos la perdida de entrenamiento
        val_loss = evaluate(
            model, criterion, val_loader, device
        )  # evaluamos el modelo en el conjunto de validacion
        epoch_val_errors.append(val_loss)  # guardamos la perdida de validacion

        if do_early_stopping:
            early_stopping(val_loss)  # llamamos al early stopping

        if log_fn is not None:  # si se pasa una funcion de log
            if (epoch + 1) % log_every == 0:  # loggeamos cada log_every epocas
                log_fn(epoch, train_loss, val_loss)  # llamamos a la funcion de log

        if do_early_stopping and early_stopping.early_stop:
            print(
                f"Detener entrenamiento en la época {epoch + 1}, la mejor pérdida fue {early_stopping.best_score:.5f}"
            )
            break

    return epoch_train_errors, epoch_val_errors


def plot_training(train_errors, val_errors):
    # Graficar los errores
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    plt.title("Training and Validation Loss")  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico


def model_classification_report(
    model,
    dataloader,
    device,
    nclasses,
    digits=2,
    do_confusion_matrix=False,
    do_balanced_accuracy=False,
    target_names=None,
):
    """
    Imprime accuracy y el reporte de clasificación (precision, recall, F1 por clase) de un modelo multiclase.

    Args:
        model (torch.nn.Module): Modelo a evaluar (su salida son logits de shape (batch, nclasses)).
        dataloader (torch.utils.data.DataLoader): Datos de evaluación.
        device (str): Dispositivo donde corre el modelo.
        nclasses (int): Cantidad de clases.
        digits (int, optional): Decimales del reporte. Por defecto 2.
        do_confusion_matrix (bool, optional): Si es True, imprime también la matriz de confusión.
        do_balanced_accuracy (bool, optional): Si es True, imprime también la precisión balanceada (promedio del recall por clase).
        target_names (list[str], optional): Nombres de las clases para el reporte. Si es None, se usan los índices 0..nclasses-1.
    """
    if target_names is None:
        target_names = [str(i) for i in range(nclasses)]

    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Reporte de clasificación
    report = classification_report(
        all_labels, all_preds, target_names=target_names, digits=digits
    )
    print("Reporte de clasificación:\n", report)

    # Matriz de confusión
    if do_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        print("Matriz de confusión:\n", cm, "\n")

    # Precisión balanceada
    if do_balanced_accuracy:
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Precisión balanceada: {bal_acc:.4f}\n")


def model_binary_classification_report(
    model, dataloader, device, threshold=0.5, logits=True, target_names=None
):
    """
    Imprime accuracy y el reporte de clasificación de un modelo binario con una única salida por muestra.

    Args:
        model (torch.nn.Module): Modelo a evaluar (salida de shape (batch, 1) o (batch,)).
        dataloader (torch.utils.data.DataLoader): Datos de evaluación.
        device (str): Dispositivo donde corre el modelo.
        threshold (float, optional): Umbral para decidir la clase positiva. Por defecto 0.5.
        logits (bool, optional): Si es True, la salida del modelo son logits y se aplica sigmoid antes del umbral.
        target_names (list[str], optional): Nombres de las dos clases. Por defecto ["Normal", "Anomalous"].
    """
    if target_names is None:
        target_names = ["Normal", "Anomalous"]

    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if logits:
                preds = (torch.sigmoid(outputs) >= threshold).long().squeeze()
            else:
                preds = (outputs >= threshold).long().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Reporte de clasificación
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("Reporte de clasificación:\n", report)


def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()

def show_tensor_image_with_mask_overlay(image_tensor, mask_tensor, title="Image with Mask Overlay", figsize=(8, 8), alpha=0.5, mask_color='Reds'):
    """
    Muestra una imagen con su máscara superpuesta.

    Args:
        image_tensor (torch.Tensor): Tensor que representa la imagen. Size (C, H, W).
        mask_tensor (torch.Tensor): Tensor que representa la máscara. Size (H, W) o (1, H, W).
        title (str, optional): Título de la imagen. Por defecto es "Image with Mask Overlay".
        figsize (tuple, optional): Tamaño de la figura. Por defecto es (8, 8).
        alpha (float, optional): Transparencia de la máscara (0=transparente, 1=opaco). Por defecto es 0.5.
        mask_color (str, optional): Colormap de la máscara. Por defecto es 'Reds'.
    """
    plt.figure(figsize=figsize)
    
    # Mostrar imagen base
    if image_tensor.shape[0] == 1:
        plt.imshow(image_tensor.squeeze(), cmap="gray")
    else:  # Assume RGB
        plt.imshow(image_tensor.permute(1, 2, 0))
        
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.squeeze(0)
    
    # Superponer la máscara
    plt.imshow(mask_tensor, cmap=mask_color, alpha=alpha * (mask_tensor > 0), vmin=0, vmax=1)
    
    plt.title(title)
    plt.axis("off")
    plt.show()
