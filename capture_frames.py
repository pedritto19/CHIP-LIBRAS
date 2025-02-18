import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Nome do gesto que estamos capturando
GESTO = input("Digite o nome do gesto a ser capturado (ex: A, B, C...): ")

# Criar arquivo CSV se não existir
csv_file = "gestures_dataset.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        header = [f"x{i}" for i in range(63)] + ["timestamp", "movement", "flipped", "label"]
        writer.writerow(header)

# Captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variáveis para detectar movimentação
last_hand_positions = None

print(f"📸 Capturando gesto '{GESTO}', pressione 'q' para parar...")

with open(csv_file, "a", newline="") as file:
    writer = csv.writer(file)

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
                    hand_data.append(lm.z)  # Profundidade

                # Calcular movimentação da mão
                movement = 0
                if last_hand_positions is not None:
                    movement = np.linalg.norm(np.array(hand_data) - np.array(last_hand_positions))

                last_hand_positions = hand_data.copy()

                # Adicionar timestamp
                timestamp = time.time()

                # Salvar dados normais
                writer.writerow(hand_data + [timestamp, movement, 0, GESTO])

                # Criar versão espelhada (flip horizontal)
                flipped_hand_data = []
                for i in range(0, len(hand_data), 3):
                    flipped_hand_data.append(1 - hand_data[i])  # Espelha apenas X
                    flipped_hand_data.append(hand_data[i+1])
                    flipped_hand_data.append(hand_data[i+2])

                # Salvar dados espelhados
                writer.writerow(flipped_hand_data + [timestamp, movement, 1, GESTO])

        cv2.imshow("Captura de Gesto", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Dados do gesto '{GESTO}' salvos com sucesso em {csv_file}!")
