import tensorflow as tf
import requests
import os
import h5py

model_url = 'https://drive.google.com/uc?id=1-32h4cyJfcWA6jHeikq5iC19qc_AY3-C&confirm=t'
output_path = 'last_model.h5'

def download_model(url, output):
    if not os.path.exists(output):
        print("Descargando el modelo desde Google Drive...")
        response = requests.get(url, stream=True)
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Modelo descargado correctamente.")
    else:
        print("El modelo ya existe en la ruta especificada.")


# Descargar el modelo
download_model(model_url, output_path)

# Cargar el modelo
try:
    model = tf.keras.models.load_model(output_path)
    print("Modelo cargado correctamente.")
except (UnicodeDecodeError, ValueError, OSError) as e:
    print(f"Error al cargar el modelo: {e}")
