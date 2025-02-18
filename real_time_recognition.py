import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time

# Carregar modelo treinado
model_file = "gestures_model.pkl"
with open(model_file, "rb") as f:
    clf = pickle.load(f)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Buffer para suavização e movimentação
frame_buffer = deque(maxlen=5)
last_hand_positions = None

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

            # Extrair pontos da mão (X, Y, Z)
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.append(lm.x)
                hand_data.append(lm.y)
                hand_data.append(lm.z)

            # Calcular movimentação
            movement = 0
            if last_hand_positions is not None:
                movement = np.linalg.norm(np.array(hand_data) - np.array(last_hand_positions))
            last_hand_positions = hand_data.copy()

            if len(hand_data) == 63:
                frame_buffer.append(hand_data + [movement])  # Adiciona movimento (agora 64 colunas)

                # Criar versão espelhada e adicionar ao buffer
                flipped_hand_data = []
                for i in range(0, len(hand_data), 3):
                    flipped_hand_data.append(1 - hand_data[i])  # Espelha apenas X
                    flipped_hand_data.append(hand_data[i+1])
                    flipped_hand_data.append(hand_data[i+2])

                frame_buffer.append(flipped_hand_data + [movement])  # Adiciona movimento para versão espelhada

                # Suavizar os dados antes da previsão
                smoothed_data = np.mean(np.array(frame_buffer, dtype=np.float32), axis=0).reshape(1, -1)

                # Certificar que o input tem exatamente 64 colunas
                if smoothed_data.shape[1] == 64:
                    # Fazer previsão
                    predicted_gesture = clf.predict(smoothed_data)[0]

                    # Mostrar resultado na tela
                    cv2.putText(frame, f"Gesto: {predicted_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🚪 Programa encerrado.")
