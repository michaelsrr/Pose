import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def calculate_angle(point1, point2):
    return np.arctan2(point2.y - point1.y, point2.x - point1.x) * 180 / np.pi

def detect_posture_in_image(image_path):
    # Cargar la imagen
    image = cv2.imread("image_0002.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(static_image_mode=True, model_complexity=0) as holistic:
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Cálculo de los ángulos o desplazamientos
            shoulders_angle = calculate_angle(left_shoulder, right_shoulder)
            hips_angle = calculate_angle(left_hip, right_hip)
            print(f"Shoulders Angle: {shoulders_angle:.2f}, Hips Angle: {hips_angle:.2f}")

            # Determinar la postura
            if abs(shoulders_angle) < 5 and abs(hips_angle) < 5:
                posture = "Derecha"
            elif shoulders_angle > 5:
                posture = "Inclinada hacia la derecha"
            elif shoulders_angle < -5:
                posture = "Inclinada hacia la izquierda"
            elif hips_angle > 5 or hips_angle < -5:
                posture = "Inclinada hacia adelante o hacia atrás"
            else:
                posture = "Postura no reconocida"

            # Mostrar la postura en la imagen
            cv2.putText(image, f"Postura: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Visualización de los landmarks (opcional)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        # Mostrar la imagen procesada
        cv2.imshow("Postura Detectada", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Ejemplo de uso
detect_posture_in_image("ruta_a_tu_imagen.jpg")
