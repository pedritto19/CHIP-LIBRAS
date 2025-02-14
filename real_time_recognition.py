import cv2
import mediapipe as mp
import numpy as np
import pickle

# Carregar modelo treinado
model_file = "gestures_model.pkl"
with open(model_file, "rb") as f:
    clf = pickle.load(f)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Captura de vídeo
cap = cv2.VideoCapture(0)

print("📷 Câmera ligada. Faça um gesto para classificar!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrair pontos da mão
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.append(lm.x)
                hand_data.append(lm.y)

            # Converter para numpy array
            hand_data = np.array(hand_data).reshape(1, -1)

            # Fazer previsão
            predicted_gesture = clf.predict(hand_data)[0]

            # Mostrar resultado na tela
            cv2.putText(frame, f"Gesto: {predicted_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🚪 Programa encerrado.")
