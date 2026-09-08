# Taller de Deep Learning

Bienvenidos al Taller de Deep Learning perteneciente a la cátedra de Inteligencia Artificial. Este repositorio está organizado para facilitar el aprendizaje y la práctica de los conceptos avanzados de Deep Learning a lo largo del año académico. Todo el material y las implementaciones se realizan utilizando PyTorch.

## Estructura del Repositorio

```
TALLER-DEEP-LEARNING-LETRA/
│
├── Entregables/
│   ├── 2024/
│   │   ├── Tarea_1-letra.ipynb
│   │   ├── Tarea_2-letra.ipynb
│   │   └── obligatorio/Obligatorio-letra.ipynb
│   └── 2025/
│       └── ...
├── assets/
│   └── imágenes usadas por las notebooks (se cargan por URL desde este repo)
├── 00 - Intro a PyTorch/
│   └── Intro_PyTorch-letra.ipynb
├── 01 - Perceptron/
│   └── Perceptron-letra.ipynb
├── 02 - California House Princing (WandB)/
│   └── California_House-letra.ipynb
├── ...
└── Extras/
    └── notebooks complementarios (Transfer Learning, ResNet, etc.)
```
## Descripción de Archivos

- **`-letra.ipynb`**: Este archivo contiene la introducción al tema de la clase, la teoría relevante y los ejercicios que deben ser completados por los estudiantes.
- **`utils.py`**: Funciones auxiliares (entrenamiento, evaluación, gráficas) usadas por los notebooks de la clase.

> Las soluciones de los ejercicios se mantienen en un repositorio privado de la cátedra.

## Lista de Clases

### 00 - Introducción a PyTorch

**Objetivos de Aprendizaje:**

1. Entender los conceptos básicos de PyTorch y su unidad de cálculo fundamental, el tensor.
2. Familiarizarse con la creación y manipulación de tensores.
3. Explorar operaciones básicas en PyTorch.

### 01 - Perceptron

**Objetivos de Aprendizaje:**

1. Implementar un perceptrón simple para resolver funciones lógicas (AND, OR, XOR).
2. Implementar un Perceptrón Multicapa (MLP) para resolver problemas más complejos.
3. Explorar la implementación opcional de otras funciones lógicas (NAND, NOR, XNOR).

### 02 - California House Princing (WandB)

**Objetivos de Aprendizaje:**

1. Implementar un modelo de regresión utilizando PyTorch.
2. Configurar y utilizar Weights & Biases (W&B) para el seguimiento de métricas y visualización de resultados.
3. Realizar preprocesamiento de datos y dividirlos en conjuntos de entrenamiento y prueba.

### 03 - MIT-BIH

**Objetivos de Aprendizaje:**

1. Implementar técnicas de regularización como early stopping y dropout.
2. Entrenar modelos de clasificación utilizando el MIT-BIH Arrhythmia Database.
3. Evaluar el impacto de las técnicas de regularización en el rendimiento del modelo.

### 04 - Intro CNN

**Objetivos de Aprendizaje:**

1. Entender los conceptos básicos de las Redes Neuronales Convolucionales (CNN).
2. Entrenar una CNN en PyTorch utilizando el dataset FashionMNIST.
3. Evaluar el rendimiento del modelo utilizando métricas adecuadas y visualización de resultados.

### 05 - CNNs Avanzadas: Transfer Learning

**Objetivos de Aprendizaje:**

1. Comparar el rendimiento de **diferentes arquitecturas avanzadas**.
2. Comparar el uso de **data augmentation**.
3. Implementar **Transfer Learning** con modelos pre-entrenados.
