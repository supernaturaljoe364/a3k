# tic_tac_toe_2p.py
# A true two-player, peer-to-peer Tic-Tac-Toe game using sockets and OpenCV.
# Both players run this same script.

import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import pickle
import struct
import threading
from collections import deque

# --- SETTINGS ---
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // 3
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 4
GESTURE_THRESHOLD = 0.06
PORT = 9999

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

board = np.zeros((3, 3), dtype=int)
turn = 1  # 1 for X, 2 for O
my_player_id = 0
conn = None
moves_queue = deque()

# --- NETWORKING ---
def network_thread(c):
    """Listens for incoming moves from the other player."""
    while True:
        try:
            data_len_packed = c.recv(8)
            if not data_len_packed: break
            msg_size = struct.unpack("Q", data_len_packed)[0]
            data = b""
            while len(data) < msg_size:
                packet = c.recv(4096)
                if not packet: break
                data += packet
            move = pickle.loads(data)
            moves_queue.append(move)
        except (ConnectionResetError, ConnectionAbortedError):
            print("Connection with the other player was lost.")
            break
        except Exception as e:
            print(f"Network error: {e}")
            break

def send_move(move):
    """Sends a move to the other player."""
    if conn:
        try:
            data = pickle.dumps(move)
            msg = struct.pack("Q", len(data)) + data
            conn.sendall(msg)
        except Exception as e:
            print(f"Failed to send move: {e}")

# --- GESTURE HELPERS ---
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]; pips = [3, 6, 10, 14, 18]; res = []
    for i, j in zip(tips, pips):
        if i == 4: res.append(hand.landmark[i].x < hand.landmark[j].x if hand.landmark[i].x < hand.landmark[0].x else hand.landmark[i].x > hand.landmark[j].x)
        else: res.append(hand.landmark[i].y < hand.landmark[j].y)
    return res

def is_two_fingers(hand): # For Player X
    fingers = fingers_up(hand)
    return fingers[1] and fingers[2] and not any([fingers[0], fingers[3], fingers[4]])

def is_ok_sign(hand): # For Player O
    dist = math.hypot(hand.landmark[4].x - hand.landmark[8].x, hand.landmark[4].y - hand.landmark[8].y)
    return dist < GESTURE_THRESHOLD

def get_cell(x, y):
    return int(y / CELL_SIZE), int(x / WINDOW_SIZE) # Corrected to use WINDOW_SIZE for y calc

# --- DRAWING ---
def draw_board(frame):
    for i in range(1, 3):
        cv2.line(frame, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), LINE_COLOR, LINE_THICKNESS)
        cv2.line(frame, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), LINE_COLOR, LINE_THICKNESS)
    for r in range(3):
        for c in range(3):
            cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
            if board[r, c] == 1:
                offset = CELL_SIZE // 4
                cv2.line(frame, (cx - offset, cy - offset), (cx + offset, cy + offset), (0, 0, 255), 4)
                cv2.line(frame, (cx + offset, cy - offset), (cx - offset, cy + offset), (0, 0, 255), 4)
            elif board[r, c] == 2:
                cv2.circle(frame, (cx, cy), CELL_SIZE // 4, (255, 0, 0), 4)

# --- MAIN ---
if __name__ == "__main__":
    role = input("Enter your role ('host' or 'client'): ").lower()
    
    if role == 'host':
        my_player_id = 1 # Host is X
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_ip = socket.gethostbyname(socket.gethostname())
        s.bind(('', PORT))
        s.listen(1)
        print(f"Hosting on IP: {host_ip}")
        print("Waiting for client to connect...")
        conn, addr = s.accept()
        print("Connected by:", addr)
    elif role == 'client':
        my_player_id = 2 # Client is O
        host_ip = input("Enter host IP: ")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host_ip, PORT))
            conn = s
            print("Connected to host.")
        except Exception as e:
            print(f"Connection failed: {e}")
            exit()
    else:
        print("Invalid role.")
        exit()

    threading.Thread(target=network_thread, args=(conn,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    last_gesture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
        
        # Process incoming moves first
        if moves_queue:
            r, c, player = moves_queue.popleft()
            if board[r, c] == 0:
                board[r, c] = player
                turn = 1 if player == 2 else 2

        # My turn gesture detection
        if turn == my_player_id:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                gesture_made = False
                if my_player_id == 1: gesture_made = is_two_fingers(hand)
                elif my_player_id == 2: gesture_made = is_ok_sign(hand)

                if gesture_made and (time.time() - last_gesture_time > 2):
                    tip = hand.landmark[12] # Use middle finger tip for pointing
                    px, py = int(tip.x * WINDOW_SIZE), int(tip.y * WINDOW_SIZE)
                    row, col = get_cell(px, py)
                    
                    if 0 <= row < 3 and 0 <= col < 3 and board[row, col] == 0:
                        board[row, col] = my_player_id
                        move = (row, col, my_player_id)
                        send_move(move)
                        turn = 2 if my_player_id == 1 else 1
                        last_gesture_time = time.time()

        draw_board(frame)
        
        # Display whose turn it is
        turn_text = f"Turn: {'X' if turn == 1 else 'O'}"
        cv2.putText(frame, turn_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        cv2.imshow(f"Tic-Tac-Toe (Player {my_player_id})", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    if conn: conn.close()
