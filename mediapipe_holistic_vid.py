import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

#cap = cv2.VideoCapture("video_0001.mp4")
cap = cv2.VideoCapture('http://192.168.1.7:4747/video')


with mp_holistic.Holistic(
     static_image_mode=False,
     model_complexity=0) as holistic:

     while True:
          ret, frame = cap.read()
          if not ret:
               break

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = holistic.process(frame_rgb)

          # Rostro
          if results.face_landmarks:
               mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))
          
          # Mano izquierda (azul)
          if results.left_hand_landmarks:
               mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
          
          # Mano derecha (verde)
          if results.right_hand_landmarks:
               mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
          
          # Postura
          if results.pose_landmarks:
               mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

          frame = cv2.flip(frame, 1)
          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
               break

cap.release()
cv2.destroyAllWindows()
