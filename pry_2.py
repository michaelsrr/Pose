import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#cap = cv2.VideoCapture('http://192.168.1.7:4747/video')
cap = cv2.VideoCapture("video_0001.mp4")

def calculate_vertical_angle(shoulder, hip):
    return np.arctan2(hip.z - shoulder.z, hip.y - shoulder.y) * 180 / np.pi

with mp_holistic.Holistic(static_image_mode=False, model_complexity=0) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Cálculo del ángulo vertical (inclinación hacia adelante o atrás) para ambos lados
            left_angle = calculate_vertical_angle(left_shoulder, left_hip)
            right_angle = calculate_vertical_angle(right_shoulder, right_hip)

            # Promedio de los ángulos izquierdo y derecho para una mejor estimación
            avg_angle = (left_angle + right_angle) / 2

            # Determinar la inclinación hacia adelante o atrás
            if avg_angle > 10:
                posture = "Inclinada hacia adelante"
            elif avg_angle < -10:
                posture = "Inclinada hacia atras"
            else:
                posture = "Postura recta"

            # Mostrar la postura en el video
            cv2.putText(frame, f"Postura: {posture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Visualización de los landmarks (opcional)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
            break

cap.release()
cv2.destroyAllWindows()
