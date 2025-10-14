import cv2, mediapipe as mp, numpy as np, math
import socket, pickle, struct, threading
from collections import deque
import time

# ---------- SETTINGS ----------
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // 3
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 4
GESTURE_THRESHOLD = 0.06
PORT = 9999
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Game board: 0=empty,1=X,2=O
board = np.zeros((3, 3), dtype=int)
turn = 1  # X starts
moves_queue = deque()  # incoming moves from client
conn = None  # client connection

# Networking thread: receive moves from client
def recv_thread(c):
    global moves_queue
    while True:
        try:
            data_len = c.recv(8)
            if not data_len: break
            msg_size = struct.unpack("Q", data_len)[0]
            data = b""
            while len(data) < msg_size:
                packet = c.recv(4096)
                if not packet: break
                data += packet
            move = pickle.loads(data)
            moves_queue.append(move)
        except: break

def send_frame(frame):
    global conn
    try:
        data = pickle.dumps(frame)
        msg = struct.pack("Q", len(data)) + data
        conn.sendall(msg)
    except: pass

# Gesture helpers
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    res = []
    for i, j in zip(tips, pips):
        if i == 4:
            res.append(hand.landmark[i].x < hand.landmark[j].x)
        else:
            res.append(hand.landmark[i].y < hand.landmark[j].y)
    return res

def is_two_fingers(hand):
    fingers = fingers_up(hand)
    return fingers[1] and fingers[2] and not any([fingers[0], fingers[3], fingers[4]])

def get_cell(x, y):
    return int(y / CELL_SIZE), int(x / CELL_SIZE)

def draw_grid(frame):
    for i in range(1, 3):
        cv2.line(frame, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), LINE_COLOR, LINE_THICKNESS)
        cv2.line(frame, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), LINE_COLOR, LINE_THICKNESS)

def draw_marks(frame):
    for r in range(3):
        for c in range(3):
            cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
            if board[r, c] == 1:
                offset = CELL_SIZE // 4
                cv2.line(frame, (cx - offset, cy - offset), (cx + offset, cy + offset), (0, 0, 255), 4)
                cv2.line(frame, (cx + offset, cy - offset), (cx - offset, cy + offset), (0, 0, 255), 4)
            elif board[r, c] == 2:
                cv2.circle(frame, (cx, cy), CELL_SIZE // 4, (255, 0, 0), 4)

# ------------------- MAIN -------------------
def main():
    global conn, turn
    # Setup server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', PORT))
    s.listen(1)
    print("Waiting for client to connect...")
    conn, addr = s.accept()
    print("Connected by:", addr)
    threading.Thread(target=recv_thread, args=(conn,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Host gesture (X) if turn==1
            if results.multi_hand_landmarks and turn == 1:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if is_two_fingers(hand):
                    tip = hand.landmark[8]
                    px, py = int(tip.x * WINDOW_SIZE), int(tip.y * WINDOW_SIZE)
                    row, col = get_cell(px, py)
                    if board[row, col] == 0:
                        board[row, col] = 1
                        turn = 2
                        move = (row, col, 1)
                        data = pickle.dumps(move)
                        conn.sendall(struct.pack("Q", len(data)) + data)
                        cv2.waitKey(300)

            # Process incoming client moves
            while moves_queue:
                r, c, m = moves_queue.popleft()
                if board[r, c] == 0:
                    board[r, c] = m
                    turn = 1

            draw_grid(frame)
            draw_marks(frame)

            # Send frame to client
            send_frame(frame)

            # Show locally
            cv2.imshow("Tic-Tac-Toe (Host)", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
