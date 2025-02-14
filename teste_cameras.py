import cv2

# Tente diferentes índices (normalmente OBS está em 1, 2 ou 3)
camera_index = 0  # Altere conforme necessário

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ajusta a largura
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Ajusta a altura

if not cap.isOpened():
    print("❌ Erro ao acessar a câmera OBS Virtual! Verifique se ela está ativada.")
    exit()

print(f"🎥 Usando OBS Virtual Camera no índice {camera_index}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erro ao capturar frame! A câmera está ativa?")
        break

    cv2.imshow("OBS Virtual Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
