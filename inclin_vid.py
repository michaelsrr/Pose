import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture('http://192.168.1.7:4747/video')

def calculate_angle(point1, point2):
    return np.arctan2(point2.y - point1.y, point2.x - point1.x) * 180 / np.pi

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

            # Cálculo de los ángulos o desplazamientos
            shoulders_angle = calculate_angle(left_shoulder, right_shoulder)
            hips_angle = calculate_angle(left_hip, right_hip)
            print(f"Shoulders Angle: {shoulders_angle:.2f}, Hips Angle: {hips_angle:.2f}")
            print(f"Left Shoulder: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f}), Right Shoulder: ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")

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

            # Mostrar la postura en el video
            cv2.putText(frame, f"Postura: {posture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Visualización de los landmarks (opcional)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        #frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
            break

cap.release()
cv2.destroyAllWindows()
