import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
import numpy as np
import requests
import os
import h5py
import contextlib

# Mostrar la versión de TensorFlow
st.write(f"TensorFlow version: {tf.__version__}")

# Descargar el modelo desde Google Drive si no existe
def download_model():
    model_url = 'https://drive.google.com/uc?id=1-06HRTv13OMkErSI2zsLP4JZffVQivIJ'
    output = 'best_model.weights.h5'
    if not os.path.exists(output):
        try:
            response = requests.get(model_url)
            with open(output, 'wb') as f:
                f.write(response.content)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")

download_model()

# Verifica si el archivo del modelo existe
modelo_path = 'best_model.weights.h5'
if not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo")
else:
    st.success("Archivo del modelo encontrado")

# Definir el modelo base InceptionV3
base_model = InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# Añadir capas de clasificación
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Cargar los pesos del modelo

# Cargar los pesos del modelo

# Cargar solo los pesos del archivo `.h5`
try:
    model.load_weights(modelo_path)
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")




# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('El modelo predice que la imagen es de un **Perro**.')
    else:
        st.success('El modelo predice que la imagen es de un **Gato**.')
