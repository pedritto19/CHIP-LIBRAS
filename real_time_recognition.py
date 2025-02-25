import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
import time
from spellchecker import SpellChecker  # Biblioteca para correção ortográfica
from rapidfuzz import fuzz, process  # Biblioteca para comparação de palavras

# Carregar modelo treinado
model_file = "gestures_model.pkl"
with open(model_file, "rb") as f:
    clf = pickle.load(f)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Inicializar correção ortográfica e dicionário de palavras comuns
spell = SpellChecker(language="pt")  # Português (troque para "en" se quiser em inglês)
common_words = list(spell.word_frequency.keys()) # Palavras comuns do dicionário

# Buffers e variáveis para controle da palavra formada
frame_buffer = deque(maxlen=5)
last_hand_positions = None
detected_gestures = []
word = []  # Palavra sendo formada
last_detected_time = time.time()
min_time_per_letter = 1.0  # Tempo mínimo que um gesto deve permanecer para ser registrado
gesture_start_time = None
previous_letter = None  # Última letra adicionada à palavra
stable_letter = None  # Letra atualmente estável
stable_letter_start_time = None  # Tempo que a letra está estável

def correct_word(word_str):
    if not word_str:
        return word_str
    word_str = word_str.lower()
    
    # Se a palavra já for conhecida, retorna ela mesma
    if word_str in common_words:
        return word_str
    
    # Tenta corrigir usando SpellChecker
    corrected = spell.correction(word_str)
    
    result = process.extractOne(word_str, common_words, scorer=fuzz.ratio)

    # Se result for None, retorna a palavra original
    if result is None:
        return word_str  

    best_match = result[0]  # Pegamos apenas a melhor correspondência
    score = result[1] if len(result) > 1 else 0  # Evita erro se o resultado for menor que esperado
    # Desempacota apenas se result for válido


    # Se a melhor sugestão for muito parecida, usamos ela
    if score > 80:  
        return best_match
    
    return corrected if corrected else word_str  # Retorna a melhor opção encontrada 

# Captura de vídeo
cap = cv2.VideoCapture(0)

print("📷 Câmera ligada. Faça gestos para formar uma palavra!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        last_detected_time = time.time()  # Atualiza o tempo da última detecção
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
                frame_buffer.append(hand_data + [movement])

                # Suavizar os dados antes da previsão
                smoothed_data = np.mean(np.array(frame_buffer, dtype=np.float32), axis=0).reshape(1, -1)

                # Fazer previsão
                predicted_gesture = clf.predict(smoothed_data)[0]

                # Verifica se a mesma letra está sendo detectada de forma estável
                if predicted_gesture == stable_letter:
                    if time.time() - stable_letter_start_time >= min_time_per_letter:
                        # Só adiciona a letra se for diferente da anterior
                        if predicted_gesture != previous_letter:
                            word.append(predicted_gesture)
                            previous_letter = predicted_gesture
                            print(f"📝 Letra adicionada: {predicted_gesture}")

                        stable_letter = None  # Resetar para detectar a próxima letra
                else:
                    stable_letter = predicted_gesture
                    stable_letter_start_time = time.time()  # Começa a contar o tempo para estabilidade

    else:
        # Se a mão sumir por mais de 1 segundo, finaliza a palavra
        if time.time() - last_detected_time > 1.0 and word:
            raw_word = "".join(word)
            corrected_word = correct_word(raw_word)  # Corrige a palavra
            print(f"✅ Palavra finalizada: {raw_word} → Correção: {corrected_word}")
            
            word.clear()  # Limpa a palavra para a próxima detecção
            previous_letter = None  # Reseta a última letra para evitar repetições
            stable_letter = None  # Reseta a estabilidade

    # Exibir a palavra formada na tela
    raw_word = "".join(word)
    corrected_word = correct_word(raw_word)
    cv2.putText(frame, f"Palavra: {corrected_word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)

    # Se pressionar "q", finaliza e exibe a palavra final
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"✅ Palavra formada: {corrected_word}")
        break

cap.release()
cv2.destroyAllWindows()
print("🚪 Programa encerrado.")
