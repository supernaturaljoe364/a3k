import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import threading
import pickle
import struct
from collections import deque
import time

# ---------- SETTINGS ----------
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // 3
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 4
GESTURE_THRESHOLD = 0.06
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Tic-Tac-Toe board (0=empty,1=X,2=O)
board = np.zeros((3,3),dtype=int)

# Networking
moves_queue = deque()  # store moves received
turn = 1  # 1=X starts, 2=O
player_mark = None  # will be set: 1 or 2

# ------------------- GESTURE FUNCTIONS -------------------
def fingers_up(hand):
    tips = [4,8,12,16,20]
    pips = [3,6,10,14,18]
    result=[]
    for i,j in zip(tips,pips):
        if i==4:
            result.append(hand.landmark[i].x < hand.landmark[j].x)
        else:
            result.append(hand.landmark[i].y < hand.landmark[j].y)
    return result

def is_ok_sign(hand):
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    d = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return d < GESTURE_THRESHOLD

def draw_grid(frame):
    for i in range(1,3):
        cv2.line(frame, (i*CELL_SIZE,0),(i*CELL_SIZE,WINDOW_SIZE),LINE_COLOR,LINE_THICKNESS)
        cv2.line(frame, (0,i*CELL_SIZE),(WINDOW_SIZE,i*CELL_SIZE),LINE_COLOR,LINE_THICKNESS)

def draw_marks(frame):
    for r in range(3):
        for c in range(3):
            cx = c*CELL_SIZE + CELL_SIZE//2
            cy = r*CELL_SIZE + CELL_SIZE//2
            if board[r,c]==1:  # X
                offset = CELL_SIZE//4
                cv2.line(frame,(cx-offset,cy-offset),(cx+offset,cy+offset),(0,0,255),4)
                cv2.line(frame,(cx+offset,cy-offset),(cx-offset,cy+offset),(0,0,255),4)
            elif board[r,c]==2:  # O
                cv2.circle(frame,(cx,cy),CELL_SIZE//4,(255,0,0),4)

def get_cell(x,y):
    col = int(x/CELL_SIZE)
    row = int(y/CELL_SIZE)
    return row,col

# ------------------- NETWORK THREADS -------------------
def server_thread(port):
    global moves_queue
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind(('',port))
    s.listen(1)
    print("Waiting for friend to connect...")
    conn, addr = s.accept()
    print("Connected by:",addr)
    while True:
        try:
            data_len = conn.recv(8)
            if not data_len: break
            msg_size = struct.unpack("Q", data_len)[0]
            data = b""
            while len(data)<msg_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            move = pickle.loads(data)
            moves_queue.append(move)
        except:
            break

def client_thread(ip,port):
    global conn
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((ip,port))
    conn = s
    print("Connected to host")
    while True:
        time.sleep(1)  # idle; sending moves happens in main

# send move
def send_move(move):
    global conn
    try:
        data = pickle.dumps(move)
        msg = struct.pack("Q",len(data))+data
        conn.sendall(msg)
    except:
        pass

# ------------------- MAIN -------------------
def main():
    global turn, player_mark, conn
    mode = input("Are you host? (y/n): ").lower()
    if mode=='y':
        player_mark = 1
        port = int(input("Enter port to host: "))
        threading.Thread(target=server_thread,args=(port,),daemon=True).start()
    else:
        player_mark = 2
        host_ip = input("Enter host IP: ")
        port = int(input("Enter host port: "))
        threading.Thread(target=client_thread,args=(host_ip,port),daemon=True).start()

    cap = cv2.VideoCapture(0)
    trail = deque(maxlen=20)
    with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame,(WINDOW_SIZE,WINDOW_SIZE))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            selected_cell = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand)
                index_tip = hand.landmark[8]
                px, py = int(index_tip.x*WINDOW_SIZE), int(index_tip.y*WINDOW_SIZE)
                row,col = get_cell(px,py)
                selected_cell = (row,col)
                cv2.circle(frame,(px,py),10,(0,255,255),-1)
                # highlight selected cell
                cv2.rectangle(frame,(col*CELL_SIZE,row*CELL_SIZE),
                              ((col+1)*CELL_SIZE,(row+1)*CELL_SIZE),(0,255,255),2)

                # Only allow move if it's your turn and cell empty
                if turn==player_mark and board[row,col]==0:
                    # X gesture
                    if fingers[1] and fingers[2] and not any([fingers[0],fingers[3],fingers[4]]) and player_mark==1:
                        board[row,col]=1
                        send_move((row,col,1))
                        turn=2
                        cv2.waitKey(300)
                    # O gesture
                    elif is_ok_sign(hand) and player_mark==2:
                        board[row,col]=2
                        send_move((row,col,2))
                        turn=1
                        cv2.waitKey(300)

            # Process incoming moves
            while moves_queue:
                move = moves_queue.popleft()
                r,c,m = move
                if board[r,c]==0:
                    board[r,c]=m
                    turn=1 if m==2 else 2

            draw_grid(frame)
            draw_marks(frame)
            cv2.imshow("2-Player Gesture Tic-Tac-Toe",frame)

            if cv2.waitKey(1) & 0xFF==27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
