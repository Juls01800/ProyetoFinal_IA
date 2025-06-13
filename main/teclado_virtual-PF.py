import cv2
import mediapipe as mp
import pygame
import time
import numpy as np
import difflib

# ----------------- Configuración básica ---------------- #
common_words = [
    "HOLA", "CASA", "COMER", "TRABAJO", "PYTHON", "TECLADO", "OJOS", "PARPADEO",
    "PANTALLA", "CAMARA", "BIENVENIDO", "COMO", "ESTAS", "QUE", "TIEMPO",
    "NOMBRE", "USUARIO", "SISTEMA", "DIA", "NOCHE", "FELIZ"
]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

pygame.init()
screen = pygame.display.set_mode((900, 400))
pygame.display.set_caption("Teclado Eye Tracking - Estilo Divertido")

# Fuente personalizada (Comic Sans o alternativa)
try:
    font = pygame.font.SysFont("Comic Sans MS", 36)
except:
    font = pygame.font.SysFont(None, 36)

# Teclado con estilo más divertido
keys = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM<-"]
key_rects = []
suggestion_rects = []

def draw_keyboard():
    """Dibuja el teclado QWERTY con colores divertidos y bordes redondeados."""
    key_rects.clear()
    colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200)]
    
    for r, row in enumerate(keys):
        for c, key in enumerate(row):
            rect = pygame.Rect(60 + c * 65, 40 + r * 75, 60, 60)
            color = colors[(r + c) % len(colors)]
            pygame.draw.rect(screen, color, rect, border_radius=10)
            pygame.draw.rect(screen, (50, 50, 50), rect, 2, border_radius=10)
            text = font.render(key, True, (0, 0, 0))
            screen.blit(text, text.get_rect(center=rect.center))
            key_rects.append((key, rect))

def calibrate(cap, samples=20):
    """Promedia la posición del landmark del iris para centrar la pantalla."""
    print("Mira al centro para calibrar…")
    coords = []
    while len(coords) < samples:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            ih, iw, _ = frame.shape
            lm = res.multi_face_landmarks[0].landmark[468]
            cx, cy = int(lm.x * iw), int(lm.y * ih)
            coords.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("Calibración", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow("Calibración")
    avg_x = np.mean([p[0] for p in coords])
    avg_y = np.mean([p[1] for p in coords])
    return avg_x, avg_y

def eye_to_screen(gx, gy, cx, cy):
    dx, dy = gx - cx, gy - cy
    return int(450 + dx * 3.0), int(180 + dy * 3.0)

def predict(prefix, words, n=3):
    return difflib.get_close_matches(prefix.upper(), words, n=n, cutoff=0.4)

# ----------------- Inicialización principal ------------- #
cap = cv2.VideoCapture(0)
calib_x, calib_y = calibrate(cap)

typed_text = ""
text_last_update = time.time()
hover_start = None
selected_item = None
clock = pygame.time.Clock()
running = True

while running:
    screen.fill((255, 245, 230))  # fondo claro cálido
    draw_keyboard()

    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    gaze_x = gaze_y = None
    if res.multi_face_landmarks:
        ih, iw, _ = frame.shape
        lm = res.multi_face_landmarks[0].landmark[468]
        gaze_x, gaze_y = int(lm.x * iw), int(lm.y * ih)

    cursor_pos = None
    if gaze_x is not None:
        cursor_pos = eye_to_screen(gaze_x, gaze_y, calib_x, calib_y)
        pygame.draw.circle(screen, (255, 100, 100), cursor_pos, 8)

    # Autocompletado visual
    last_word = typed_text.split()[-1] if typed_text else ""
    predictions = predict(last_word, common_words, n=3)
    suggestion_rects.clear()
    for i, sug in enumerate(predictions):
        rect = pygame.Rect(50 + i * 280, 320, 260, 45)
        pygame.draw.rect(screen, (180, 240, 255), rect, border_radius=10)
        pygame.draw.rect(screen, (0, 120, 150), rect, 2, border_radius=10)
        text = font.render(sug, True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=rect.center))
        suggestion_rects.append((sug, rect))

    # Hover y dwell time
    hit_item = None
    if cursor_pos:
        for key, rect in key_rects:
            if rect.collidepoint(cursor_pos):
                pygame.draw.rect(screen, (0, 255, 0), rect, 3, border_radius=10)
                hit_item = ("key", key)
                break
        if not hit_item:
            for sug, rect in suggestion_rects:
                if rect.collidepoint(cursor_pos):
                    pygame.draw.rect(screen, (0, 200, 0), rect, 3, border_radius=10)
                    hit_item = ("sug", sug)
                    break

    if hit_item:
        if selected_item == hit_item:
            if time.time() - hover_start > 1.5:
                kind, value = hit_item
                if kind == "key":
                    typed_text = typed_text[:-1] if value == "<-" else typed_text + value
                else:
                    parts = typed_text.split()
                    if parts:
                        parts[-1] = value
                    else:
                        parts.append(value)
                    typed_text = " ".join(parts) + " "
                text_last_update = time.time()
                hover_start = None
                selected_item = None
        else:
            selected_item = hit_item
            hover_start = time.time()
    else:
        selected_item = hover_start = None

    # Mostrar texto escrito
    text_surface = font.render("Texto: " + typed_text, True, (0, 0, 0))
    screen.blit(text_surface, (50, 260))

    if typed_text and time.time() - text_last_update > 5:
        typed_text = ""

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
cv2.destroyAllWindows()
