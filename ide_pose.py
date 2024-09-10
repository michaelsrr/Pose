import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Función para determinar la postura
def classify_pose(landmarks, width, height):
    # Coordenadas de los puntos clave
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]

    # Calcular distancias y ángulos
    posture = "Unknown"
    confidence = 0

    # De pie
    if (euclidean_distance(hip_left, knee_left) > height * 0.1 and
        euclidean_distance(hip_right, knee_right) > height * 0.1 and
        abs(shoulder_left[1] - hip_left[1]) > height * 0.15 and
        abs(shoulder_right[1] - hip_right[1]) > height * 0.15):
        posture = "De pie"
        confidence = 90

    # Sentado
    if (abs(knee_left[1] - hip_left[1]) < height * 0.1 and
        abs(knee_right[1] - hip_right[1]) < height * 0.1 and
        shoulder_left[1] > hip_left[1] and shoulder_right[1] > hip_right[1]):
        posture = "Sentado"
        confidence = 85

    # Inclinación hacia adelante
    if (shoulder_left[1] > hip_left[1] and shoulder_right[1] > hip_right[1] and
        shoulder_left[1] < knee_left[1] and shoulder_right[1] < knee_right[1]):
        posture = "Inclinación hacia adelante"
        confidence = 80

    # Brazo levantado
    if (elbow_left[1] < shoulder_left[1] or elbow_right[1] < shoulder_right[1]):
        posture = "Brazo levantado"
        confidence = 95

    return posture, confidence

#Scap = cv2.VideoCapture("video_0001.mp4")
cap = cv2.VideoCapture('http://192.168.1.7:4747/video')

with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            posture, confidence = classify_pose(results.pose_landmarks.landmark, width, height)
            cv2.putText(frame, f'Postura: {posture} ({confidence}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
