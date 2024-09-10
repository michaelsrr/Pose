import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 1. Cargar y Preprocesar las Imágenes
def cargar_imagenes(ruta_dataset, img_size=(128, 128)):
    clases = ['de_pie', 'sentado', 'inclinacion', 'brazo_levantado']
    X, y = [], []
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_dataset, clase)
        for img_name in os.listdir(ruta_clase):
            img_path = os.path.join(ruta_clase, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(clase)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

ruta_dataset = 'dataset'
X, y = cargar_imagenes(ruta_dataset)

# Normalizar las imágenes
X = X / 255.0

# Convertir etiquetas a formato one-hot
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definir el Modelo CNN
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# 3. Compilar el Modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Entrenar el Modelo
history = modelo.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# 5. Evaluar el Modelo
loss, accuracy = modelo.evaluate(X_val, y_val)
print(f"Precisión en el conjunto de validación: {accuracy * 100:.2f}%")

# Guardar el modelo
modelo.save('modelo_posturas.h5')
