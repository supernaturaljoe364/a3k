# ttt_net.py
import cv2, mediapipe as mp, numpy as np, math
import socket, pickle, struct, threading, time
from collections import deque

# ---------- SETTINGS ----------
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // 3
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 4
PORT = 9999
FRAME_QUALITY = 60      # JPEG quality (0-100)
FRAME_INTERVAL = 0.15   # seconds between sending frames (~6-7 FPS)
GESTURE_COOLDOWN = 1.2  # seconds between gesture moves
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# shared game state
board = np.zeros((3, 3), dtype=int)
turn = 1         # 1 = X host starts, 2 = O
my_player_id = 0
winner = ''
conn = None

# networking buffers (thread-safe-ish queues)
moves_queue = deque()
peer_frame = None
peer_frame_lock = threading.Lock()

# --- utilities: encode/decode message dicts over TCP ---
def send_msg(sock, obj):
    """Pickle and send a python object with length prefix"""
    try:
        data = pickle.dumps(obj)
        header = struct.pack("Q", len(data))
        sock.sendall(header + data)
    except Exception as e:
        # connection problems handled by main loop
        print("send_msg error:", e)

def recv_msg(sock):
    """Receive one pickled object (blocking)"""
    header = sock.recv(8)
    if not header:
        return None
    msg_len = struct.unpack("Q", header)[0]
    data = b""
    while len(data) < msg_len:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    if not data:
        return None
    return pickle.loads(data)

# --- networking thread ---
def network_recv_thread(sock):
    global moves_queue, peer_frame
    while True:
        try:
            msg = recv_msg(sock)
            if msg is None:
                print("Connection closed by peer.")
                break
            if isinstance(msg, dict) and msg.get("type") == "frame":
                # msg['data'] is JPEG bytes
                jpg = msg["data"]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with peer_frame_lock:
                        peer_frame = frame
            elif isinstance(msg, dict) and msg.get("type") == "move":
                moves_queue.append(msg["move"])  # (r,c,player)
            else:
                # unknown message
                pass
        except Exception as e:
            print("Network receive error:", e)
            break

# --- gesture helpers ---
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    res = []
    # thumb by x
    res.append(hand.landmark[tips[0]].x < hand.landmark[pips[0]].x)
    # other fingers by y
    for i in range(1, 5):
        res.append(hand.landmark[tips[i]].y < hand.landmark[pips[i]].y)
    return res

def is_open_palm(hand):
    return all(fingers_up(hand))

def is_fist(hand):
    return not any(fingers_up(hand))

def get_cell(x, y):
    return int(y / CELL_SIZE), int(x / CELL_SIZE)

def check_win(current_board):
    for r in range(3):
        if current_board[r,0] == current_board[r,1] == current_board[r,2] and current_board[r,0] != 0:
            return current_board[r,0]
    for c in range(3):
        if current_board[0,c] == current_board[1,c] == current_board[2,c] and current_board[0,c] != 0:
            return current_board[0,c]
    if current_board[0,0] == current_board[1,1] == current_board[2,2] and current_board[0,0] != 0:
        return current_board[0,0]
    if current_board[0,2] == current_board[1,1] == current_board[2,0] and current_board[0,2] != 0:
        return current_board[0,2]
    if 0 not in current_board:
        return 'draw'
    return ''

# drawing
def draw_board_overlay(base_frame):
    frame = base_frame.copy()
    # draw grid
    for i in range(1,3):
        cv2.line(frame, (i*CELL_SIZE,0), (i*CELL_SIZE, WINDOW_SIZE), LINE_COLOR, LINE_THICKNESS)
        cv2.line(frame, (0,i*CELL_SIZE), (WINDOW_SIZE, i*CELL_SIZE), LINE_COLOR, LINE_THICKNESS)
    # draw marks
    for r in range(3):
        for c in range(3):
            cx, cy = c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2
            if board[r,c] == 1:
                off = CELL_SIZE//4
                cv2.line(frame, (cx-off, cy-off), (cx+off, cy+off), (0,0,255), 4)
                cv2.line(frame, (cx+off, cy-off), (cx-off, cy+off), (0,0,255), 4)
            elif board[r,c] == 2:
                cv2.circle(frame, (cx,cy), CELL_SIZE//4, (255,0,0), 4)
    return frame

# main
def main():
    global conn, my_player_id, turn, winner

    role = input("Enter role ('host' or 'client'): ").strip().lower()
    if role not in ("host","client"):
        print("Invalid role")
        return

    # setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if role == "host":
        my_player_id = 1
        host_ip = socket.gethostbyname(socket.gethostname())
        try:
            sock.bind(('', PORT))
            sock.listen(1)
            print("Hosting on", host_ip, "port", PORT)
            conn_socket, addr = sock.accept()
            conn = conn_socket
            print("Client connected:", addr)
        except Exception as e:
            print("Host socket error:", e)
            return
    else:
        my_player_id = 2
        host_ip = input("Enter host IP: ").strip()
        try:
            sock.connect((host_ip, PORT))
            conn = sock
            print("Connected to host.")
        except Exception as e:
            print("Client socket error:", e)
            return

    # start receive thread
    threading.Thread(target=network_recv_thread, args=(conn,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    last_frame_time = 0
    last_gesture_time = 0

    # small self preview window settings
    preview_size = 160

    while True:
        # handle incoming move messages
        while moves_queue:
            r,c,p = moves_queue.popleft()
            if 0 <= r < 3 and 0 <= c < 3 and board[r,c] == 0:
                board[r,c] = p
                turn = 1 if p == 2 else 2
                winner = check_win(board)

        ret, local_frame = cap.read()
        if not ret:
            print("Camera not accessible")
            break
        local_frame = cv2.flip(local_frame, 1)
        local_frame = cv2.resize(local_frame, (WINDOW_SIZE, WINDOW_SIZE))

        # send frame periodically
        now = time.time()
        if now - last_frame_time > FRAME_INTERVAL:
            # encode current local_frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY]
            success, jpg = cv2.imencode('.jpg', local_frame, encode_param)
            if success:
                try:
                    send_msg(conn, {"type":"frame", "data": jpg.tobytes()})
                except Exception as e:
                    print("send frame error:", e)
            last_frame_time = now

        # detect gestures only when it's my turn
        if turn == my_player_id and winner == '':
            rgb = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(local_frame, hand, mp_hands.HAND_CONNECTIONS)

                gesture = False
                if my_player_id == 1:
                    gesture = is_open_palm(hand)
                else:
                    gesture = is_fist(hand)

                if gesture and (now - last_gesture_time) > GESTURE_COOLDOWN:
                    # use index tip as pointer
                    tip = hand.landmark[8]
                    px, py = int(tip.x * WINDOW_SIZE), int(tip.y * WINDOW_SIZE)
                    r,c = get_cell(px, py)
                    if 0 <= r < 3 and 0 <= c < 3 and board[r,c] == 0:
                        board[r,c] = my_player_id
                        winner = check_win(board)
                        # send move to peer
                        send_msg(conn, {"type":"move", "move": (r,c,my_player_id)})
                        turn = 2 if my_player_id == 1 else 1
                        last_gesture_time = now

        # prepare display frame:
        display_base = None
        with peer_frame_lock:
            if peer_frame is not None:
                # show peer frame as background (resize to window)
                display_base = cv2.resize(peer_frame, (WINDOW_SIZE, WINDOW_SIZE))
        if display_base is None:
            # fallback: show a dark background if no peer frame yet
            display_base = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

        # overlay board and marks onto display_base
        display = draw_board_overlay(display_base)

        # small self preview at top-left
        small = cv2.resize(local_frame, (preview_size, preview_size))
        # put a border
        small = cv2.copyMakeBorder(small, 2,2,2,2, cv2.BORDER_CONSTANT, value=(200,200,200))
        display[10:10+preview_size+4, 10:10+preview_size+4] = small

        # show turn / winner
        status = f"Turn: {'X' if turn==1 else 'O'}"
        if winner:
            status = "Winner: " + ("X" if winner==1 else "O" if winner==2 else "Draw")
        cv2.putText(display, status, (20, WINDOW_SIZE-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        cv2.imshow(f"Tic-Tac-Toe (You are {'X' if my_player_id==1 else 'O'})", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        conn.close()
    except:
        pass

if __name__ == "__main__":
    main()
