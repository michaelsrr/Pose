import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

with mp_holistic.Holistic(
     static_image_mode=True,
     model_complexity=2) as holistic:

     image = cv2.imread("image_0004.jpg")
     
     if image is None:
         print("Error: No se pudo cargar la imagen. Verifique la ruta del archivo.")
     else:
         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         results = holistic.process(image_rgb)
         
         # rostro
         if results.face_landmarks:
             mp_drawing.draw_landmarks(
                  image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                  mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))
         
         # Mano izquierda (azul)
         if results.left_hand_landmarks:
             mp_drawing.draw_landmarks(
                  image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                  mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
         
         # Mano derecha (verde)
         if results.right_hand_landmarks:
             mp_drawing.draw_landmarks(
                  image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                  mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
         
         # Postura
         if results.pose_landmarks:
             mp_drawing.draw_landmarks(
                  image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                  mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

         cv2.imshow("Image", image)

         # Plot: puntos de referencia y conexiones en matplotlib 3D
         if results.pose_world_landmarks:
             mp_drawing.plot_landmarks(
                  results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
         cv2.waitKey(0)
     cv2.destroyAllWindows()
