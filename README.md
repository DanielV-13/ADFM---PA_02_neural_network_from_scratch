<div align="center">

# 🧠 Write your own DNN in Python: Edición extendida y comentada

### Red neuronal construida **desde cero con NumPy** para clasificar casas: 🌉 San Francisco vs. 🗽 Nueva York

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielV-13/ADFM---PA_02_neural_network_from_scratch/blob/main/ADFM_PA_02_neural_network_from_scratch.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Idioma](https://img.shields.io/badge/idioma-espa%C3%B1ol-red)

**Análisis de Datos en Física Moderna (ADFM)**, Universidad de Oviedo · 2026

<img width="900" alt="Red neuronal: visión general" src="https://github.com/user-attachments/assets/59f15f71-055b-415b-97f0-cc8f4a2d7404" />

</div>

---

## 📑 Tabla de contenidos

1. [Sobre el proyecto](#-sobre-el-proyecto)
2. [El problema: casas, señal y fondo](#-el-problema-casas-señal-y-fondo)
3. [Arquitectura de la red](#-arquitectura-de-la-red)
4. [El camino del cuaderno](#%EF%B8%8F-el-camino-del-cuaderno)
5. [Cómo aprende la red: el ciclo de 4 pasos](#-cómo-aprende-la-red-el-ciclo-de-4-pasos)
6. [Resultados](#-resultados)
7. [Experimentos propios](#-experimentos-propios)
8. [Lo que aprendí](#-lo-que-aprendí)
9. [Galería visual](#%EF%B8%8F-galería-visual)
10. [Cómo ejecutarlo](#%EF%B8%8F-cómo-ejecutarlo)
11. [Estructura del repositorio](#-estructura-del-repositorio)
12. [Créditos y licencias](#-créditos-y-licencias)

---

## 📌 Sobre el proyecto

Este repositorio contiene mi versión **extendida, comentada y explicada en español** de la práctica de aula *PA_02: Write your own DNN in python*, creada por el profesor **Pietro Vischia** para la asignatura ADFM.

El objetivo de la práctica **no es solo predecir, sino entender**. Por eso se programa cada pieza de una red neuronal a mano, sin PyTorch ni TensorFlow: la multiplicación de matrices, las funciones de activación, la función de pérdida, las derivadas, la actualización de pesos y el bucle de entrenamiento.

### ✍️ ¿Qué añadí sobre el cuaderno original?

| Añadido | Descripción |
| :--- | :--- |
| 🧭 **Brief, mapa del camino y glosario de siglas** | Una introducción que explica el propósito, la ruta lógica del cuaderno y el significado de cada sigla. |
| 🧹 **Teoría de preparación de datos** | *Outliers*, valores faltantes y escalado (MinMax, estandarización, normalización), conectado con las diapositivas de clase. |
| 💬 **Comentarios línea por línea** | Todo el código está comentado de forma granular, en español. |
| 📐 **Derivaciones matemáticas completas** | La BCE desde la teoría de la información, su versión estable con *logits* y la cancelación que da $\frac{\partial L}{\partial Z_2} = \frac{P-y}{m}$. |
| 🖼️ **20 diagramas explicativos** | Imágenes resumen de cada concepto: arquitectura, dropout, esquemas, ROC/AUC, etc. |
| ⚛️ **Analogía con física de partículas** | Clasificar casas ↔ separar **señal** de **fondo** en el CERN. |
| 🔬 **Respuestas a la discusión y al *Endgame*** | Respuestas razonadas a todas las preguntas del cuaderno. |
| 🧪 **5 experimentos propios** | Dropout, *learning rate*, sobreajuste, divergencia y entrenamiento de 200 épocas. |
| 🏁 **Conclusión general** | Un cierre que recorre todo el viaje y resume lo aprendido. |

---

## 🏙️ El problema: casas, señal y fondo

Usamos el dataset **Housing** de [R2D3](https://github.com/jadeyee/r2d3-part-1-data): casas de San Francisco y Nueva York descritas por **7 características**.

`beds` · `bath` · `price` · `year_built` · `sqft` · `price_per_sqft` · `elevation`

La red debe responder: ***"¿Qué tan probable es que esta casa sea de San Francisco?"*** Es un problema de **clasificación binaria**.

| Clase | Etiqueta | En física de partículas |
| :--- | :---: | :--- |
| 🌉 San Francisco | **1** (positiva) | **Señal**: la partícula que buscamos |
| 🗽 Nueva York | **0** (negativa) | **Fondo**: el ruido que la imita |

<div align="center">
<img width="850" alt="Machine Learning en física de partículas" src="https://github.com/user-attachments/assets/d308ed40-d0df-47a4-8651-6b72e7776f17" />
</div>

### 🗂️ División de los datos (estratificada)

| Conjunto | Casas | Papel | Analogía |
| :--- | :---: | :--- | :--- |
| 🏋️ **Train** | 275 (60%) | La red **aprende** aquí (ajusta pesos) | Ejercicios del libro |
| 🧭 **Validation** | 92 (20%) | **Decidir** configuraciones y guardar el mejor modelo | Exámenes parciales |
| 🔒 **Test** | 92 (20%) | **Medición final**, una sola vez | Examen final |

> ⚠️ El escalador `MinMaxScaler` se ajusta **solo con Train** para evitar *data leakage*, y las etiquetas nunca se escalan.

---

## 📐 Arquitectura de la red

<div align="center">
<img width="850" alt="Diagrama de arquitectura de la red neuronal" src="https://github.com/user-attachments/assets/9cde1bac-a934-4878-b450-5e4d5c94149c" />
</div>

| Capa | Tamaño | Activación | Parámetros |
| :--- | :---: | :--- | :---: |
| Entrada | 7 características | — | — |
| Oculta | 20 neuronas | **ReLU** (+ dropout opcional) | 7×20 + 20 = **160** |
| Salida | 1 neurona | **Sigmoide** → probabilidad de SF | 20×1 + 1 = **21** |
| | | **Total** | **181** |

$$Z_1 = XW_1 + b_1 \;\rightarrow\; A_1 = \text{ReLU}(Z_1) \;\rightarrow\; Z_2 = A_1W_2 + b_2 \;\rightarrow\; P = \sigma(Z_2)$$

<div align="center">
<img width="750" alt="Funciones de activación ReLU y Sigmoide" src="https://github.com/user-attachments/assets/9103e390-89b1-41b8-aef4-dce20c61e6fd" />
</div>

---

## 🗺️ El camino del cuaderno

| # | Etapa | ¿Qué pregunta responde? |
| :---: | :--- | :--- |
| 0 | 🧹 Preparación y limpieza de datos | ¿Cómo dejo los datos "digeribles" para la red? |
| 1 | 📊 Carga, deduplicación, división y escalado | ¿Con qué datos aprende y con cuáles se evalúa? |
| 2 | 📐 La red como conjunto de matrices | ¿Qué forma tiene la red por dentro? |
| Ej. 1 | ⚡ Funciones de activación | ¿Cómo "decide" cada neurona? |
| Ej. 2 | ➡️ *Forward pass* | ¿Cómo hace la red una predicción? |
| — | 📉 Función de pérdida (BCE) | ¿Qué tan mal lo hizo? |
| Ej. 3 | ⬅️ *Backpropagation* | ¿Quién causó el error? |
| Ej. 4 | 🔧 Descenso del gradiente | ¿Cómo corrijo el error? |
| — | ✅ *Gradient checking* | ¿Mis derivadas son correctas? |
| Ej. 5 | 📦 Stochastic / Batch / Minibatch | ¿Cada cuánto aprende la red? |
| Ej. 6 | 🎭 *Inverted dropout* | ¿Cómo evito que memorice? |
| — | 🚂 *Training loop* + *checkpointing* | ¿Cómo aprende de verdad? |
| — | 🔬 Comparación 3 esquemas × 2 dropouts × 3 semillas | ¿Qué configuración funciona mejor? |
| 8 | 🔒 Congelar elección y evaluar en Test | ¿Qué tan bueno es con datos nunca vistos? |
| 9 | 🧪 Experimentos libres #1 a #5 | ¿Qué pasa si cambio las reglas del juego? |
| 🎯 | *Endgame* y conclusión | ¿Qué me llevo de todo esto? |

---

## 🔁 Cómo aprende la red: el ciclo de 4 pasos

| Paso | Nombre técnico | Qué hace | Analogía del estudiante |
| :---: | :--- | :--- | :--- |
| 1 | ➡️ **Forward pass** | Calcula una probabilidad a partir de las 7 características | Intenta resolver el ejercicio |
| 2 | 📉 **Pérdida (BCE)** | Mide el error; castiga fuerte los errores cometidos con mucha seguridad | Revisa el solucionario |
| 3 | ⬅️ **Backpropagation** | Reparte la "culpa" del error entre los 181 pesos (sin modificarlos) | Detecta dónde falló su razonamiento |
| 4 | 🔧 **Descenso del gradiente** | $\theta \leftarrow \theta - \eta \nabla L$ | Corrige su forma de estudiar |

<div align="center">
<img width="800" alt="Loop de entrenamiento" src="https://github.com/user-attachments/assets/d1c297a7-3143-47be-8384-b2b632bd5b1b" />
</div>

### 📦 Los tres esquemas de actualización

| Esquema | Filas por actualización | Actualizaciones por época |
| :--- | :---: | :---: |
| `stochastic` | 1 | $N$ |
| `batch` | $N$ | 1 |
| `minibatch` | 32 | $\lceil N/32 \rceil$ |

<div align="center">
<img width="700" alt="Épocas y esquemas" src="https://github.com/user-attachments/assets/2461a5c5-dee9-4f1f-b030-1c3d87cc973b" />
</div>

### 🎭 Inverted dropout

Durante el entrenamiento se apagan neuronas ocultas al azar con probabilidad $r$ y las que quedan se amplifican por $1/q$ (con $q = 1 - r$). La **misma máscara** se reutiliza en el *backward pass*. En validación y test, el dropout está **apagado**.

<div align="center">
<img width="800" alt="Dropout esquema general" src="https://github.com/user-attachments/assets/6abe0f88-ebdd-425e-adfc-0640651ba3c0" />
</div>

---

## 📊 Resultados

### 🔬 Comparación de esquemas (120 épocas, *learning rate* 0.05, 3 semillas, sin dropout)

| Esquema | Actualizaciones totales | BCE de validación final ↓ | Observación |
| :--- | :---: | :---: | :--- |
| `stochastic` | 33 000 | **≈ 0.249** | Aprende muy rápido por época, pero es el más lento en tiempo real |
| `minibatch` (32) | 1 080 | ≈ 0.311 | Curvas suaves y el mejor equilibrio velocidad/estabilidad |
| `batch` | 120 | ≈ 0.609 | Seguía bajando: le faltó presupuesto de entrenamiento |

> 🎭 **Dropout (0.25):** no mejoró de forma consistente la BCE en esta red pequeña, que no sufría sobreajuste. La regularización es un seguro, **no una mejora garantizada**.

<div align="center">
<img width="900" alt="Curvas de aprendizaje por esquema" src="https://github.com/user-attachments/assets/020ba185-7169-456e-82bb-a59aa83a2b52" />
</div>

### 🔒 Evaluación final en Test (modelo congelado)

**Configuración elegida por validación:** `stochastic`, sin dropout, checkpoint de la época 114.

| Métrica | Valor |
| :--- | :---: |
| Línea base (clase mayoritaria) | ~53.8% |
| **Accuracy en Test** | **93.5%** (86 / 92) |
| **ROC AUC** | **0.948** |
| BCE en Test | 0.361 |
| Matriz de confusión | TN = 39 · TP = 47 · FP = 3 · FN = 3 |

<div align="center">
<img width="900" alt="Curva ROC, matriz de confusión y AUC explicadas" src="https://github.com/user-attachments/assets/33856160-e36a-48d5-b871-4dd9352b1439" />
</div>

---

## 🧪 Experimentos propios

Todos se hicieron con el esquema `minibatch`, la semilla `SEEDS[0]` y **sin tocar el conjunto de Test**.

| # | Qué se varió | Qué se mantuvo fijo | Conclusión |
| :---: | :--- | :--- | :--- |
| **1** | Dropout: 0, 0.1, 0.25, 0.5 | Lote 32, 40 épocas, LR 0.05 | Sin dropout se logra la menor BCE; 0.5 "asfixia" la capacidad de aprender. |
| **2** | *Learning rate*: 0.001, 0.01, 0.05, 0.5 | Lote 32, 40 épocas, sin dropout | LR muy bajos se estancan (subajuste); 0.5 baja rápido pero oscila. |
| **3** | Dropout 0 vs. 0.5, con **80 épocas** | Lote 32, LR 0.05 | Sin regularización desciende mejor; con 0.5 aprende más lento. Sin rebote de sobreajuste visible. |
| **4** | LR extremos: 0.05, 0.5, 1.5, 3.0 | Lote 16, 30 épocas | LR altísimos provocan curvas caóticas y un error persistentemente alto. |
| **5** | Configuraciones extremas con **200 épocas** | Semilla | Un lote pequeño sin regularización se estabiliza en la BCE más baja; dropout 0.7 genera ruido persistente. |

<div align="center">
<img width="700" alt="Guía de experimentación" src="https://github.com/user-attachments/assets/092a7c00-b808-41af-835a-c2da30406f7f" />
</div>

---

## 📚 Lo que aprendí

- 🗂️ **Train aprende, Validation decide, Test mide una sola vez.** Mezclarlos invalida los resultados.
- 📐 Una red neuronal no es magia: son **multiplicaciones de matrices** seguidas de funciones de activación.
- ⚡ Sin la no-linealidad de la **ReLU**, apilar capas equivale a tener una sola.
- ✨ Con Sigmoide + BCE, el gradiente de salida se simplifica a **"predicción menos realidad"**: $(P-y)/m$.
- ⬅️ El **backpropagation** es la regla de la cadena aplicada a matrices, capa por capa.
- 🔧 El ***learning rate*** tiene un punto medio: muy pequeño no avanza y muy grande rebota.
- 📦 **Minibatch** equilibra frecuencia de aprendizaje y eficiencia de cómputo.
- 🎭 El **dropout** es un seguro contra la memorización, no una mejora garantizada.
- 📊 **BCE y accuracy pueden ordenar los modelos de forma distinta**; la ROC/AUC evalúa todos los umbrales a la vez.
- 🧪 El Deep Learning es empírico: **hay que leer los resultados reales**, no asumir quién ganará.

---

## 🖼️ Galería visual

<details>
<summary><b>Haz clic para ver todos los diagramas explicativos del cuaderno</b></summary>

<br>

**Preparación de datos**
<img width="800" alt="Limpieza de datos" src="https://github.com/user-attachments/assets/8253f367-d18c-4d11-aaa4-e871b8b4c4e2" />

**Learning rate**
<img width="800" alt="Learning rate imagen explicatoria" src="https://github.com/user-attachments/assets/955727f6-67c9-4dbf-a366-008493cab572" />

**Loop de entrenamiento (II)**
<img width="800" alt="Loop de entrenamiento imagen 2" src="https://github.com/user-attachments/assets/1432ab9c-a543-42af-b98d-1cd5d654430d" />

**Dropout**
<img width="800" alt="Dropout imagen y explicación" src="https://github.com/user-attachments/assets/a9235f7a-ea32-48c8-88b6-cd51b9236b41" />
<img width="800" alt="Dropout imagen simplificada" src="https://github.com/user-attachments/assets/862a2009-e894-48f0-a02c-36ba41c73a65" />
<img width="800" alt="Escalado del dropout" src="https://github.com/user-attachments/assets/815801cd-7e10-43c1-ab51-e71c57e3f7fc" />

**Diagnóstico de curvas de aprendizaje**
<img width="800" alt="Diagnóstico de las learning curves" src="https://github.com/user-attachments/assets/0d1f29c0-49c3-40c5-b8b5-4cf7435bfb6a" />

**Métricas de rendimiento**
<img width="800" alt="Métricas de performance ROC curve (español)" src="https://github.com/user-attachments/assets/bb54808d-35a6-4936-bb54-a7980d52ea39" />
<img width="800" alt="Métricas de performance ROC curve" src="https://github.com/user-attachments/assets/b6469a52-2c42-40ff-a8c3-84723f9ec47c" />
<img width="800" alt="Curva ROC y AUC" src="https://github.com/user-attachments/assets/80703c4a-e256-42f8-9d1b-b179fc6bd6b8" />

</details>

---

## ▶️ Cómo ejecutarlo

### Opción 1: Google Colab (recomendada)
Haz clic en el botón [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielV-13/ADFM---PA_02_neural_network_from_scratch/blob/main/ADFM_PA_02_neural_network_from_scratch.ipynb) y ejecuta **Runtime → Run all**.

> 💡 El cuaderno pesa unos 17 MB porque lleva las imágenes embebidas. Si GitHub no lo previsualiza, ábrelo en Colab o en [nbviewer](https://nbviewer.org/github/DanielV-13/ADFM---PA_02_neural_network_from_scratch/blob/main/ADFM_PA_02_neural_network_from_scratch.ipynb).

### Opción 2: Local
```bash
git clone https://github.com/DanielV-13/ADFM---PA_02_neural_network_from_scratch.git
cd ADFM---PA_02_neural_network_from_scratch
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook ADFM_PA_02_neural_network_from_scratch.ipynb
```

📦 **No hay que descargar datos:** el CSV viene embebido en el cuaderno (comprimido con gzip + base64 y verificado con SHA-256).

🎛️ **Usar tus propias implementaciones:** añade el número de ejercicio a `SOLVED_EXERCISES` (por ejemplo `{1, 2}`) en la celda de configuración y ejecuta *Restart & Run All*. Con el conjunto vacío se usan las soluciones de referencia.

---

## 📁 Estructura del repositorio

```
ADFM---PA_02_neural_network_from_scratch/
├── ADFM_PA_02_neural_network_from_scratch.ipynb   # Cuaderno completo (código + explicaciones + imágenes)
├── *.jpg                                          # Diagramas explicativos usados en el cuaderno
└── README.md                                      # Este archivo
```

---

## 🙏 Créditos y licencias

| | |
| :--- | :--- |
| 📘 **Cuaderno original** | *Write your own DNN in python* © **Pietro Vischia**, 2026 · [theory lecture](https://www.hep.uniovi.es/vischia/persistent/2026-09to12_ADFM_2_vischia.html) |
| ✍️ **Edición extendida y comentada** | © **Daniel Vaca Velastegui**, 2026 |
| 🤖 **Asistencia** | **Gemini Flash** (Google) y **Claude Opus 5.5** (Anthropic): explicaciones, comentarios de código, diagramas y revisión final |
| 🏠 **Datos** | [R2D3 Housing data](https://github.com/jadeyee/r2d3-part-1-data), bajo licencia [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Datos educativos: según sus autores, no deben usarse para sacar conclusiones sobre ninguno de los dos mercados inmobiliarios. |
| 📄 **Referencia** | Srivastava et al. (2014), [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html) |
| 📄 **Referencia** | scikit-learn, [*Common pitfalls: preprocessing and data leakage*](https://scikit-learn.org/stable/common_pitfalls.html) |

---

<div align="center">

**Daniel Vaca Velastegui** · Estudiante de intercambio en la Universidad de Oviedo · ADFM 2026

*"Las redes que se usan en el CERN o en la industria son más grandes, pero funcionan exactamente con estas mismas ideas."*

</div>
