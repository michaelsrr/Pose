import os
import cv2

# URL de la cámara del celular
url_camara = 'http://192.168.1.7:4747/video'
cap = cv2.VideoCapture(url_camara)

# Asegúrate de que el directorio del dataset exista
def crear_directorios(ruta_dataset):
    clases = ['de_pie', 'sentado', 'inclinacion', 'brazo_levantado']
    for clase in clases:
        ruta_clase = os.path.join(ruta_dataset, clase)
        if not os.path.exists(ruta_clase):
            os.makedirs(ruta_clase)

# Función para capturar imágenes desde el stream de video
def capturar_imagenes(ruta_dataset, clase, num_imagenes=100, img_size=(224, 224)):
    crear_directorios(ruta_dataset)
    ruta_clase = os.path.join(ruta_dataset, clase)
    
    contador = 0
    while contador < num_imagenes:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video")
            break
        
        # Mostrar el frame en la ventana
        cv2.imshow('Captura de imagenes', frame)

        # Redimensionar la imagen al tamaño adecuado (224x224 para más detalles de cuerpo completo)
        img_resized = cv2.resize(frame, img_size)

        # Guardar la imagen en la carpeta correspondiente
        img_path = os.path.join(ruta_clase, f"{clase}_{contador}.jpg")
        cv2.imwrite(img_path, img_resized)
        
        contador += 1
        print(f"Imagen {contador}/{num_imagenes} capturada para la clase '{clase}'")
        
        # Presiona 'q' para salir de la captura
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"Captura de imágenes de la clase '{clase}' completada.")

# Ruta para guardar las imágenes
ruta_dataset = r'C:\Users\Michael\Documents\GitHub\Pose\dataset'

# Ingresar las posturas que vas a capturar
clases = ['de_pie', 'sentado', 'inclinacion', 'brazo_levantado']

for clase in clases:
    print(f"Capturando imágenes para la clase '{clase}'. Presiona 'q' para detener.")
    capturar_imagenes(ruta_dataset, clase, num_imagenes=50)  # Cambia el número según las imágenes que quieras capturar

cap.release()
cv2.destroyAllWindows()
