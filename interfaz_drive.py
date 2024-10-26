import streamlit as st
import tensorflow as tf
import requests
import os

# Descargar el modelo desde Google Drive si no existe
def download_model():
    model_url = 'https://drive.google.com/uc?id=1--93G0ZMb2uxjQtIS0ZoNUCLDM7AkWH0'  # Reemplazar con el ID de tu modelo
    output = 'best_model.keras'
    if not os.path.exists(output):
        try:
            response = requests.get(model_url)
            with open(output, 'wb') as f:
                f.write(response.content)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")

download_model()

# Ruta completa al archivo .keras
modelo_path = 'best_model.keras'

# Cargar el modelo completo
try:
    model = tf.keras.models.load_model(modelo_path)
    st.success("Modelo cargado correctamente.")
except (UnicodeDecodeError, ValueError, OSError) as e:
    st.error(f"Error al cargar el modelo: {e}")
    model = None

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen de un gato o un perro...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('El modelo predice que la imagen es de un **Perro**.')
    else:
        st.success('El modelo predice que la imagen es de un **Gato**.')
