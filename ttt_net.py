import cv2, mediapipe as mp, numpy as np, socket, struct, pickle, threading, time

PORT = 9999
FRAME_SIZE = (320, 240)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

board = np.zeros((3, 3), dtype=int)
turn, my_id = 1, 0
conn = None
winner = ''
last_move_time = 0
CELL_SIZE, WIN_SIZE = 200, 600

def check_win(b):
    for r in range(3):
        if b[r,0]==b[r,1]==b[r,2]!=0: return b[r,0]
    for c in range(3):
        if b[0,c]==b[1,c]==b[2,c]!=0: return b[0,c]
    if b[0,0]==b[1,1]==b[2,2]!=0: return b[0,0]
    if b[0,2]==b[1,1]==b[2,0]!=0: return b[0,2]
    return 'draw' if 0 not in b else ''

def draw_board(frame):
    for i in range(1,3):
        cv2.line(frame,(i*CELL_SIZE,0),(i*CELL_SIZE,WIN_SIZE),(255,255,255),2)
        cv2.line(frame,(0,i*CELL_SIZE),(WIN_SIZE,i*CELL_SIZE),(255,255,255),2)
    for r in range(3):
        for c in range(3):
            cx, cy = c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2
            if board[r,c]==1:
                off=CELL_SIZE//4
                cv2.line(frame,(cx-off,cy-off),(cx+off,cy+off),(0,0,255),3)
                cv2.line(frame,(cx+off,cy-off),(cx-off,cy+off),(0,0,255),3)
            elif board[r,c]==2:
                cv2.circle(frame,(cx,cy),CELL_SIZE//4,(255,0,0),3)

def send_data(sock, data):
    pkt = pickle.dumps(data)
    sock.sendall(struct.pack("Q", len(pkt)) + pkt)

def recv_data(sock):
    data_len = struct.unpack("Q", sock.recv(8))[0]
    data = b''
    while len(data) < data_len:
        packet = sock.recv(4096)
        if not packet: return None
        data += packet
    return pickle.loads(data)

def send_video(sock, cap):
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(cv2.resize(frame, FRAME_SIZE), 1)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        send_data(sock, buffer)
    cap.release()

def recv_video(sock, q):
    while True:
        try:
            buf = recv_data(sock)
            if buf is None: break
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            q.append(frame)
            if len(q)>1: q.pop(0)
        except: break

if __name__ == "__main__":
    role = input("Enter role (host/client): ").strip().lower()
    if role=='host':
        my_id=1
        s=socket.socket(); s.bind(('',PORT)); s.listen(1)
        print("Waiting for client...")
        conn,addr=s.accept(); print("Connected:",addr)
    else:
        my_id=2
        host=input("Enter ngrok host (ex: 0.tcp.in.ngrok.io): ")
        port=int(input("Enter ngrok port: "))
        s=socket.socket(); s.connect((host,port)); conn=s; print("Connected to host.")

    # start video threads
    cap=cv2.VideoCapture(0)
    frames=[]
    threading.Thread(target=recv_video,args=(conn,frames),daemon=True).start()
    threading.Thread(target=send_video,args=(conn,cap),daemon=True).start()

    while True:
        if not frames: continue
        remote_frame=cv2.resize(frames[0],FRAME_SIZE)
        ret, local_frame=cv2.VideoCapture(0).read()
        if not ret: break
        local_frame=cv2.flip(cv2.resize(local_frame,FRAME_SIZE),1)
        both=np.hstack((local_frame,remote_frame))
        board_overlay=np.zeros((WIN_SIZE,WIN_SIZE,3),dtype=np.uint8)
        draw_board(board_overlay)
        board_small=cv2.resize(board_overlay,(both.shape[1],both.shape[0]))
        display=cv2.addWeighted(both,0.8,board_small,0.4,0)
        cv2.imshow("Gesture TicTacToe",display)
        if cv2.waitKey(1)&0xFF==27: break

    cv2.destroyAllWindows()
