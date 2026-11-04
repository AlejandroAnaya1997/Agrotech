# 🌿 AgroTech: Clasificación de Enfermedades en Cultivos mediante MobileNetV2

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Licencia](https://img.shields.io/badge/Estado-Desplegado-brightgreen?style=for-the-badge)

Aplicación web interactiva basada en **Visión por Computador** y **Deep Learning** para el diagnóstico foliar automatizado de 38 tipos de cultivos (sanos y enfermos), desarrollada como proyecto de evaluación para la asignatura de **Inteligencia Artificial (2026)**.

🌐 **Demo en vivo en Streamlit Cloud:** [agrotech2026.streamlit.app](https://agrotech2026.streamlit.app)

---

## 📌 1. Metodología CRISP-DM Aplicada

El desarrollo del proyecto siguió las 6 fases del ciclo de vida CRISP-DM enfocado a soluciones agrícolas (AgroTech):

1. **Comprensión del Negocio:** Facilitar a agricultores y técnicos agrícolas una herramienta rápida y accesible desde dispositivos móviles para detectar patologías foliares en etapas tempranas.
2. **Comprensión de los Datos:** Análisis exploratorio de las **87,000+ imágenes** del conjunto de datos *New Plant Diseases Dataset* distribuidas en **38 clases** distintas.
3. **Preparación de Datos:** Resizing a $224 \times 224 \text{ px}$, normalización en rango $[-1, 1]$ alineado con MobileNetV2, data augmentation (rotación, flip horizontal/vertical, zoom) y split estratificado (Train / Validation / Test).
4. **Modelado:** Transfer Learning inicial congelando el *backbone* de **MobileNetV2** preentrenado en ImageNet, seguido de una etapa de *Fine-Tuning* descongelando las últimas capas convolucionales con un *Learning Rate* reducido ($\eta = 10^{-5}$).
5. **Evaluación:** Evaluación de la matriz de confusión, precisión, recall y F1-Score general del modelo en el conjunto de prueba.
6. **Despliegue:** Construcción de una interfaz responsiva en **Streamlit** desplegada en **Streamlit Community Cloud** con soporte HTTPS para cámara en vivo desde celulares.

---

## 🏗️ 2. Arquitectura de la Solución

* **Red Neuronal Base:** `MobileNetV2` (arquitectura liviana optimizada para ejecución en CPU/móviles mediante convoluciones separables en profundidad).
* **Capas Superiores (Head Customizado):**
  * `GlobalAveragePooling2D`
  * `BatchNormalization`
  * `Dense(256, activation='relu')`
  * `Dropout(0.4)`
  * `Dense(38, activation='softmax')`

---

## 📂 3. Estructura del Repositorio

```text
agrotech/
├── app.py                  # Código principal de la aplicación web Streamlit
├── requirements.txt        # Dependencias de Python fijadas para producción
├── runtime.txt             # Definición del entorno (Python 3.11)
├── .python-version         # Forzar versión de Python en entornos de despliegue
├── README.md               # Documentación del proyecto
└── modelo/
    ├── modelo.keras        # Pesos y arquitectura del modelo entrenado
    └── clases.json         # Mapeo de índices a los nombres de las 38 clases



🚀 4. Instalación y Ejecución Local
Prerrequisitos
Python 3.11 instalado.

Git.

Pasos
Clonar el repositorio:

Bash
git clone [https://github.com/TU_USUARIO/agrotech-plant-disease.git](https://github.com/TU_USUARIO/agrotech.git)
cd agrotech

Instalar dependencias:

Bash
python -m pip install -r requirements.txt
Ejecutar la aplicación web localmente:

Bash
python -m streamlit run app.py
Abre en tu navegador la dirección indicada: http://localhost:8501.

📱 5. Características de la Aplicación
📷 Carga por Archivo o Cámara en Vivo: Permite subir imágenes (PNG, JPG, JPEG) o capturar la foto de la hoja directamente con la cámara del celular.

📊 Predicción y Nivel de Confianza: Muestra la enfermedad detectada o estado de salud de la planta con su porcentaje de probabilidad.

🏆 Top 3 de Diagnósticos: Desglosa las 3 clases con mayor certeza predichas por la red neuronal.
