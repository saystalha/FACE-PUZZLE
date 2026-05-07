import cv2
import numpy as np
import random

# =====================================
# MediaPipe Imports
# =====================================
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils

# =====================================
# Initialize Detectors
# =====================================
face_detector = face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# Camera
# =====================================
cap = cv2.VideoCapture(0)

# =====================================
# Puzzle Settings (2 x 2)
# =====================================
ROWS = 2
COLS = 2
PUZZLE_W = 400
PUZZLE_H = 400
TILE_W = PUZZLE_W // COLS
TILE_H = PUZZLE_H // ROWS

tiles = []
original_tiles = []
puzzle_ready = False
face_saved = False
puzzle_solved = False

# =====================================
# Selection State
# =====================================
selected_tile = None
hover_tile = None
hold_counter = 0
HOLD_THRESHOLD = 20        # ~0.6s at 30fps to select a tile

thumb_hold_counter = 0
THUMB_HOLD_THRESHOLD = 20  # ~0.6s to confirm thumb-up capture

thumb_down_counter = 0     # frames thumb has been held down for reset

# =====================================
# Thumb Up Detection
# =====================================
def is_thumb_up(hand_landmarks):
    """
    True if thumb is extended upward and all other fingers are folded.
    MediaPipe y increases downward, so 'above' = smaller y value.
    """
    lm = hand_landmarks.landmark
    thumb_up    = (lm[4].y < lm[3].y) and (lm[3].y < lm[2].y)
    index_down  = lm[8].y  > lm[6].y
    middle_down = lm[12].y > lm[10].y
    ring_down   = lm[16].y > lm[14].y
    pinky_down  = lm[20].y > lm[18].y
    return thumb_up and index_down and middle_down and ring_down and pinky_down

# =====================================
# Thumb Down Detection
# =====================================
def is_thumb_down(hand_landmarks):
    """
    True if thumb is pointing downward and all other fingers are folded.
    Thumb tip must be BELOW the MCP joint (larger y = lower on screen).
    """
    lm = hand_landmarks.landmark
    thumb_down  = (lm[4].y > lm[3].y) and (lm[3].y > lm[2].y)
    index_down  = lm[8].y  > lm[6].y
    middle_down = lm[12].y > lm[10].y
    ring_down   = lm[16].y > lm[14].y
    pinky_down  = lm[20].y > lm[18].y
    return thumb_down and index_down and middle_down and ring_down and pinky_down

# =====================================
# Create Puzzle Tiles
# =====================================
def create_puzzle(img):
    parts = []
    for r in range(ROWS):
        for c in range(COLS):
            tile = img[r*TILE_H:(r+1)*TILE_H, c*TILE_W:(c+1)*TILE_W]
            parts.append(tile)
    return parts

# =====================================
# Shuffle Puzzle
# =====================================
def shuffle_tiles(parts):
    temp = parts.copy()
    for _ in range(30):
        a = random.randint(0, len(temp)-1)
        b = random.randint(0, len(temp)-1)
        temp[a], temp[b] = temp[b], temp[a]
    return temp

# =====================================
# Get Tile Index from Finger Position
# =====================================
def get_tile_from_finger(fx, fy, frame_w, frame_h):
    col = int((fx / frame_w) * COLS)
    row = int((fy / frame_h) * ROWS)
    col = max(0, min(COLS - 1, col))
    row = max(0, min(ROWS - 1, row))
    return row * COLS + col

# =====================================
# Draw Puzzle with Highlights
# =====================================
def show_puzzle(parts, hover_idx=None, selected_idx=None, hold_pct=0.0, solved=False):
    board_tiles = []
    for i, tile in enumerate(parts):
        t = tile.copy()

        if i == selected_idx:
            cv2.rectangle(t, (4, 4), (TILE_W-4, TILE_H-4), (0, 255, 0), 6)
            cv2.putText(t, "SELECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif i == hover_idx and selected_idx is None:
            cv2.rectangle(t, (4, 4), (TILE_W-4, TILE_H-4), (255, 150, 0), 4)
            bar_w = int((TILE_W - 8) * hold_pct)
            cv2.rectangle(t, (4, TILE_H-18), (4 + bar_w, TILE_H-6), (255, 200, 0), -1)

        elif i == hover_idx and selected_idx is not None:
            cv2.rectangle(t, (4, 4), (TILE_W-4, TILE_H-4), (0, 220, 255), 4)
            cv2.putText(t, "SWAP?", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        board_tiles.append(t)

    row1 = np.hstack(board_tiles[0:2])
    row2 = np.hstack(board_tiles[2:4])
    board = np.vstack([row1, row2])

    cv2.line(board, (TILE_W, 0), (TILE_W, PUZZLE_H), (255, 255, 255), 2)
    cv2.line(board, (0, TILE_H), (PUZZLE_W, TILE_H), (255, 255, 255), 2)

    if solved:
        overlay = board.copy()
        cv2.rectangle(overlay, (0, 0), (PUZZLE_W, PUZZLE_H), (0, 200, 80), -1)
        cv2.addWeighted(overlay, 0.35, board, 0.65, 0, board)
        cv2.putText(board, "SOLVED!", (60, PUZZLE_H // 2 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 5)
        cv2.putText(board, "SOLVED!", (60, PUZZLE_H // 2 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 220, 80), 3)
        cv2.putText(board, "👎 Thumb Down to restart", (35, PUZZLE_H // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Puzzle", board)

# =====================================
# Check Solved
# =====================================
def is_solved():
    for i in range(4):
        if not np.array_equal(tiles[i], original_tiles[i]):
            return False
    return True

# =====================================
# Main Loop
# =====================================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # =====================================
    # Face Detection — Whole Face
    # =====================================
    face_results = face_detector.process(rgb)
    face_crop = None

    if face_results.multi_face_landmarks and not face_saved:
        for face_landmarks in face_results.multi_face_landmarks:
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

            x_min = max(0, min(x_coords) - 40)
            y_min = max(0, min(y_coords) - 60)
            x_max = min(w, max(x_coords) + 40)
            y_max = min(h, max(y_coords) + 60)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size != 0:
                cv2.imshow("Face Preview", face_crop)

    # =====================================
    # Keyboard Input
    # =====================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # =====================================
    # Hand Detection — always active
    # =====================================
    hand_results = hand_detector.process(rgb)
    current_hover = None

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)

            # ---- THUMB UP → Capture face (only before puzzle starts) ----
            if not face_saved:
                if is_thumb_up(hand_landmarks):
                    thumb_hold_counter += 1

                    bar_w = int(w * 0.4 * min(thumb_hold_counter / THUMB_HOLD_THRESHOLD, 1.0))
                    cv2.rectangle(frame,
                                  (w//2 - int(w*0.2), h - 30),
                                  (w//2 - int(w*0.2) + bar_w, h - 10),
                                  (0, 200, 255), -1)
                    cv2.putText(frame, "Hold thumb up to capture...",
                                (w//2 - 160, h - 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

                    if thumb_hold_counter >= THUMB_HOLD_THRESHOLD:
                        if face_crop is not None and face_crop.size != 0:
                            face_img = cv2.resize(face_crop, (PUZZLE_W, PUZZLE_H))
                            cv2.imwrite("face.jpg", face_img)
                            original_tiles = create_puzzle(face_img)
                            tiles = shuffle_tiles(original_tiles)
                            face_saved = True
                            puzzle_ready = True
                            puzzle_solved = False
                            selected_tile = None
                            hold_counter = 0
                            thumb_hold_counter = 0
                            print("👍 Thumb Up! Face Puzzle Created!")
                else:
                    thumb_hold_counter = max(0, thumb_hold_counter - 1)

            # ---- THUMB DOWN → Reset puzzle (only after puzzle starts) ----
            if puzzle_ready:
                if is_thumb_down(hand_landmarks):
                    thumb_down_counter += 1

                    bar_w = int(w * 0.4 * min(thumb_down_counter / THUMB_HOLD_THRESHOLD, 1.0))
                    cv2.rectangle(frame,
                                  (w//2 - int(w*0.2), h - 30),
                                  (w//2 - int(w*0.2) + bar_w, h - 10),
                                  (0, 80, 255), -1)
                    cv2.putText(frame, "Hold thumb down to reset...",
                                (w//2 - 160, h - 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

                    if thumb_down_counter >= THUMB_HOLD_THRESHOLD:
                        tiles = shuffle_tiles(original_tiles)
                        selected_tile = None
                        hold_counter = 0
                        puzzle_solved = False
                        thumb_down_counter = 0
                        hover_tile = None
                        print("👎 Thumb Down! Puzzle Reset!")
                else:
                    thumb_down_counter = max(0, thumb_down_counter - 1)

            # ---- INDEX FINGER → Tile pointing (only after puzzle starts) ----
            if puzzle_ready and not puzzle_solved:
                index_tip = hand_landmarks.landmark[8]
                fx = int(index_tip.x * w)
                fy = int(index_tip.y * h)

                cv2.circle(frame, (fx, fy), 12, (0, 100, 255), -1)
                cv2.circle(frame, (fx, fy), 12, (255, 255, 255), 2)

                current_hover = get_tile_from_finger(fx, fy, w, h)
                col = current_hover % COLS
                row_idx = current_hover // COLS
                cv2.putText(frame, f"Tile [{row_idx},{col}]", (fx + 15, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # =====================================
    # Hold-to-Select Logic
    # =====================================
    if puzzle_ready and not puzzle_solved:
        if current_hover is not None:
            if current_hover == hover_tile:
                hold_counter += 1
            else:
                hover_tile = current_hover
                hold_counter = 0

            if hold_counter >= HOLD_THRESHOLD:
                if selected_tile is None:
                    selected_tile = current_hover
                    hold_counter = 0
                    print(f"Tile {selected_tile} selected!")
                elif current_hover != selected_tile:
                    tiles[selected_tile], tiles[current_hover] = \
                        tiles[current_hover], tiles[selected_tile]
                    print(f"Swapped tile {selected_tile} <-> {current_hover}")
                    selected_tile = None
                    hold_counter = 0
                else:
                    selected_tile = None
                    hold_counter = 0
                    print("Deselected.")
        else:
            hold_counter = max(0, hold_counter - 2)

    # =====================================
    # Check Solved
    # =====================================
    if puzzle_ready and not puzzle_solved and is_solved():
        puzzle_solved = True
        selected_tile = None
        hold_counter = 0
        hover_tile = None
        print("🎉 PUZZLE SOLVED!")

    # =====================================
    # Show Puzzle
    # =====================================
    if puzzle_ready:
        hold_pct = min(hold_counter / HOLD_THRESHOLD, 1.0)
        show_puzzle(
            tiles,
            hover_idx=None if puzzle_solved else hover_tile,
            selected_idx=None if puzzle_solved else selected_tile,
            hold_pct=0.0 if puzzle_solved else hold_pct,
            solved=puzzle_solved
        )

        if puzzle_solved:
            cv2.putText(frame, "PUZZLE SOLVED!", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

    # =====================================
    # HUD
    # =====================================
    if not puzzle_ready:
        cv2.putText(frame, "👍 Thumbs Up to capture your face!",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    elif puzzle_solved:
        cv2.putText(frame, "Solved! 👎 Thumb Down to restart",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    elif selected_tile is not None:
        cv2.putText(frame, f"Tile {selected_tile} selected — point at another to swap",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Hold finger on a tile to select it",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.putText(frame, "Q=Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.imshow("Face Puzzle Game", frame)

    if key == ord('q'):
        break

# =====================================
# Cleanup
# =====================================
cap.release()
cv2.destroyAllWindows()