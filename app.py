import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configuración de página de Streamlit
st.set_page_config(
    page_title="AgroTech - Clasificación de Enfermedades Foliares",
    page_icon="🌿",
    layout="centered"
)

TAMANO_IMAGEN = (224, 224)

@st.cache_resource
def cargar_modelo_y_clases():
    """
    Carga el modelo Keras y los nombres de las clases en memoria una sola vez
    para evitar demoras en las iteraciones de la app.
    """
    modelo = keras.models.load_model("modelo/modelo.keras")
    with open("modelo/clases.json", "r", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases

def preprocesar_imagen(imagen_pil):
    """
    Aplica exactamente el mismo preprocesamiento usado durante el entrenamiento:
    conversión a RGB, redimensionamiento a 224x224 y formato float32 con lote (1, 224, 224, 3).
    """
    imagen_rgb = imagen_pil.convert("RGB").resize(TAMANO_IMAGEN)
    arreglo = np.array(imagen_rgb, dtype="float32")
    arreglo_batch = np.expand_dims(arreglo, axis=0)
    return arreglo_batch

# Cargar artefactos
try:
    modelo, clases = cargar_modelo_y_clases()
except Exception as error:
    st.error(f"Error al cargar los artefactos del modelo en 'modelo/': {error}")
    st.stop()

# Interfaz Principal
st.title("🌿 AgroTech - Diagnóstico Agrícola Automatizado")
st.markdown("""
Bienvenido al sistema inteligente de diagnóstico de patologías foliares. 
Suba una imagen de la hoja o capture una foto desde su dispositivo móvil para obtener un diagnóstico en tiempo real.
""")

st.write("---")

metodo_entrada = st.radio(
    "Seleccione el método de entrada de imagen:",
    ("Subir imagen desde archivo", "Tomar foto con la cámara"),
    horizontal=True
)

archivo_imagen = None
if metodo_entrada == "Subir imagen desde archivo":
    archivo_imagen = st.file_uploader("Cargue una imagen foliar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
else:
    archivo_imagen = st.camera_input("Tome una foto clara de la hoja del cultivo")

if archivo_imagen is not None:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Fotografía cargada", width=350)
    
    if st.button("🔍 Diagnosticar Estado de Salud", type="primary"):
        with st.spinner("Analizando patrones convolucionales..."):
            tensor_entrada = preprocesar_imagen(imagen)
            predicciones = modelo.predict(tensor_entrada, verbose=0)[0]
            
            idx_max = int(np.argmax(predicciones))
            clase_predicha = clases[idx_max]
            confianza = float(predicciones[idx_max]) * 100
            
            # Limpiar nombre de clase para la interfaz
            nombre_limpio = clase_predicha.replace("___", " - ").replace("_", " ")
            
            st.success(f"### Diagnóstico Predicho:\n**{nombre_limpio}**")
            st.metric(label="Nivel de Confianza", value=f"{confianza:.2f} %")
            
            st.write("---")
            st.subheader("📊 Top 3 de Probabilidades:")
            
            indices_top3 = np.argsort(predicciones)[::-1][:3]
            for i in indices_top3:
                nombre = clases[i].replace("___", " - ").replace("_", " ")
                prob = float(predicciones[i]) * 100
                st.write(f"**{nombre}**: `{prob:.2f}%`")
                st.progress(float(predicciones[i] / 100.0))