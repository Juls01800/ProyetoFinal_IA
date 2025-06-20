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
screen = pygame.display.set_mode((800, 350))
pygame.display.set_caption("Teclado Virtual con Eye Tracking + Botón Sugerencias")
font = pygame.font.SysFont(None, 40)

keys = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM<-"]
key_rects = []                           # teclas
suggestion_rects = []                    # botones de sugerencias

# ----------------- Funciones utilitarias ---------------- #
def draw_keyboard():
    """Dibuja el teclado QWERTY."""
    key_rects.clear()
    for r, row in enumerate(keys):
        for c, key in enumerate(row):
            rect = pygame.Rect(50 + c * 70, 50 + r * 70, 60, 60)
            pygame.draw.rect(screen, (200, 200, 200), rect)
            screen.blit(font.render(key, True, (0, 0, 0)),
                        (rect.x + 15, rect.y + 15))
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
    """Mapea la mirada (coordenadas de la cámara) a la ventana Pygame."""
    dx, dy = gx - cx, gy - cy
    return int(400 + dx * 3.0), int(150 + dy * 3.0)

def predict(prefix, words, n=3):
    return difflib.get_close_matches(prefix.upper(), words, n=n, cutoff=0.4)

# ----------------- Inicialización principal ------------- #
cap = cv2.VideoCapture(0)
calib_x, calib_y = calibrate(cap)

typed_text = ""
text_last_update = time.time()           # para autolimpieza
hover_start = None                       # temporizador dwell
selected_item = None                     # tecla o sugerencia actual
clock = pygame.time.Clock()
running = True

while running:
    screen.fill((30, 30, 30))
    draw_keyboard()

    # --------- Cámara y posición de mirada -------- #
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
        pygame.draw.circle(screen, (255, 0, 0), cursor_pos, 5)

    # --------- Autocompletado y botones ------------ #
    last_word = typed_text.split()[-1] if typed_text else ""
    predictions = predict(last_word, common_words, n=3)
    suggestion_rects.clear()
    for i, sug in enumerate(predictions):
        rect = pygame.Rect(50 + i * 240, 300, 220, 40)   # tres botones iguales
        pygame.draw.rect(screen, (100, 255, 100), rect)
        screen.blit(font.render(sug, True, (0, 0, 0)),
                    (rect.x + 10, rect.y + 5))
        suggestion_rects.append((sug, rect))

    # --------- Detección de hover / dwell ---------- #
    hit_item = None
    if cursor_pos:
        # Primero teclas
        for key, rect in key_rects:
            if rect.collidepoint(cursor_pos):
                pygame.draw.rect(screen, (0, 255, 0), rect, 3)
                hit_item = ("key", key)
                break
        # Luego sugerencias
        if not hit_item:
            for sug, rect in suggestion_rects:
                if rect.collidepoint(cursor_pos):
                    pygame.draw.rect(screen, (0, 200, 0), rect, 3)
                    hit_item = ("sug", sug)
                    break

    # Lógica de dwell time (1.5 s)
    if hit_item:
        if selected_item == hit_item:
            if time.time() - hover_start > 1.5:
                kind, value = hit_item
                if kind == "key":
                    if value == "<-":
                        typed_text = typed_text[:-1]
                    else:
                        typed_text += value
                else:  # “sug”
                    # reemplaza la última palabra por la sugerencia y añade espacio
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

    # -------------- Dibujar texto ------------------ #
    screen.blit(font.render("Texto: " + typed_text, True, (255, 255, 255)),
                (50, 250))

    # -------------- Autolimpieza ------------------- #
    if typed_text and time.time() - text_last_update > 5:
        typed_text = ""

    # -------------- Eventos de Pygame -------------- #
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

# -------------- Fin de programa ------------------- #
cap.release()
pygame.quit()
cv2.destroyAllWindows()
