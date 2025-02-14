import cv2
import mediapipe as mp
import csv
import os

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
        header = [f"x{i}" for i in range(42)] + ["label"]
        writer.writerow(header)

# Captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ajusta a largura
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Ajusta a altura


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

                # Extrair pontos da mão
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append(lm.x)
                    hand_data.append(lm.y)
                
                hand_data.append(GESTO)  # Adiciona o rótulo do gesto
                writer.writerow(hand_data)

        cv2.imshow("Captura de Gesto", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Dados do gesto '{GESTO}' salvos com sucesso em {csv_file}!")
