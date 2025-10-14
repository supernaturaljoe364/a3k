import cv2, mediapipe as mp, socket, pickle, struct, threading, math
from collections import deque

WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE//3
GESTURE_THRESHOLD = 0.06
PORT = 9999

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

moves_queue = deque()
conn=None

# ------------------- NETWORK -------------------
def recv_thread(c):
    global moves_queue
    while True:
        try:
            # receive frame
            data_len = c.recv(8)
            if not data_len: break
            msg_size = struct.unpack("Q", data_len)[0]
            data = b""
            while len(data)<msg_size:
                packet = c.recv(4096)
                if not packet: break
                data += packet
            frame = pickle.loads(data)
            moves_queue.append(frame)
        except: break

def send_move(c, move):
    try:
        data = pickle.dumps(move)
        c.sendall(struct.pack("Q",len(data))+data)
    except: pass

def fingers_up(hand):
    tips=[4,8,12,16,20];pips=[3,6,10,14,18];res=[]
    for i,j in zip(tips,pips):
        if i==4: res.append(hand.landmark[i].x<hand.landmark[j].x)
        else: res.append(hand.landmark[i].y<hand.landmark[j].y)
    return res

def is_ok_sign(hand):
    d = math.hypot(hand.landmark[4].x - hand.landmark[8].x,
                   hand.landmark[4].y - hand.landmark[8].y)
    return d<GESTURE_THRESHOLD

def get_cell(x,y): return int(y/CELL_SIZE), int(x/CELL_SIZE)

# ------------------- MAIN -------------------
def main():
    global conn
    host_ip = input("Enter host IP: ")
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((host_ip,PORT))
    conn=s
    threading.Thread(target=recv_thread,args=(conn,),daemon=True).start()

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame,(WINDOW_SIZE,WINDOW_SIZE))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            selected_cell=None

            # Capture client gesture (O) and send to host
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand)
                index_tip = hand.landmark[8]
                px,py = int(index_tip.x*WINDOW_SIZE), int(index_tip.y*WINDOW_SIZE)
                row,col = get_cell(px,py)
                selected_cell=(row,col)
                cv2.rectangle(frame,(col*CELL_SIZE,row*CELL_SIZE),
                              ((col+1)*CELL_SIZE,(row+1)*CELL_SIZE),(0,255,255),2)
                if is_ok_sign(hand):
                    move=(row,col,2)
                    send_move(conn,move)
                    cv2.waitKey(300)

            # Display latest frame from host
            if moves_queue:
                display_frame = moves_queue.pop()
                moves_queue.clear()
                cv2.imshow("Tic-Tac-Toe (Client)",display_frame)
            else:
                # fallback: show own camera
                cv2.imshow("Tic-Tac-Toe (Client)",frame)

            if cv2.waitKey(1)&0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__=="__main__":
    main()