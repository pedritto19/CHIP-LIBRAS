import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time
from spellchecker import SpellChecker
from rapidfuzz import fuzz, process
import json

model_file = "gestures_model.pkl"
with open(model_file, "rb") as f:
    clf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

spell = SpellChecker(language="pt")
common_words = list(spell.word_frequency.keys())

frame_buffer = deque(maxlen=5)
last_hand_positions = None
word = []
last_detected_time = time.time()
min_time_per_letter = 1.0
previous_letter = None
stable_letter = None
stable_letter_start_time = None

def correct_word(word_str):
    if not word_str:
        return word_str
    word_str = word_str.lower()
    if word_str in common_words:
        return word_str
    corrected = spell.correction(word_str)
    result = process.extractOne(word_str, common_words, scorer=fuzz.ratio)
    if result and result[1] > 80:
        return result[0]
    return corrected if corrected else word_str

def save_result(raw_word, corrected_word):
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump({"raw_word": raw_word, "corrected_word": corrected_word}, f)

cap = cv2.VideoCapture(0)

word = []
previous_letter = stable_letter = None
stable_letter_start_time = last_detected_time = time.time()
min_time_per_letter = 1.0

print("📷 Câmera ligada. Faça gestos para formar uma palavra!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        last_detected_time = time.time()
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_data = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            movement = np.linalg.norm(np.array(hand_data) - np.array(last_hand_positions)) if last_hand_positions else 0
            last_hand_positions = hand_data.copy()

            if len(hand_data) == 63:
                frame_buffer.append(hand_data + [movement])
                smoothed_data = np.mean(np.array(frame_buffer, dtype=np.float32), axis=0).reshape(1, -1)
                predicted_gesture = clf.predict(smoothed_data)[0]

                if predicted_gesture == stable_letter:
                    if time.time() - stable_letter_start_time >= min_time_per_letter:
                        if predicted_gesture != previous_letter:
                            word.append(predicted_gesture)
                            previous_letter = predicted_gesture
                            print(f"📝 Letra adicionada: {predicted_gesture}")
                            save_result("".join(word), correct_word("".join(word)))
                        stable_letter = None
                else:
                    stable_letter = predicted_gesture
                    stable_letter_start_time = time.time()

            last_hand_positions = hand_data.copy()

    else:
        if time.time() - last_detected_time > 1.0 and word:
            raw_word = "".join(word)
            corrected_word = correct_word(raw_word)
            print(f"✅ Palavra finalizada: {raw_word} → Correção: {corrected_word}")
            save_result(raw_word, corrected_word)
            word.clear()
            previous_letter = stable_letter = None

    cv2.imshow("Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
