# tic_tac_toe_firebase.py
# A two-player, gesture-controlled Tic-Tac-Toe game using a central Firebase server.
# Both players run this exact same script and connect using a Game ID.
# CONTROLS: Player X = Open Palm | Player O = Fist



#note: required, firebase-admin
# make sure protobuf in version 4.25.3 to run both mediapipe and firebase
# --- Step 1: Import Libraries ---
import cv2
import mediapipe as mp
import firebase_admin
from firebase_admin import credentials, firestore
import time
import math
import webbrowser
import threading
import random

# --- Step 2: Configuration ---
PINCH_THRESHOLD = 0.08 
CELL_SIZE = 150
BOARD_COLOR = (255, 255, 255)
SYMBOL_COLOR_X = (0, 0, 255) # Red for X
SYMBOL_COLOR_O = (255, 0, 0) # Blue for O
HIGHLIGHT_COLOR = (0, 255, 0)

# --- Step 3: Firebase Initialization ---
try:
    cred = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Successfully connected to Firebase.")
except Exception as e:
    print(f"Error connecting to Firebase: {e}")
    print("Please ensure 'credentials.json' is in the correct folder and you've followed the setup instructions.")
    exit()

# --- Step 4: Player & Game Setup ---
MY_SYMBOL = ''
GAME_ID = ''
game_ref = None

# --- Step 5: Initialization ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

game_state = {}
state_lock = threading.Lock()
last_move_time = 0

# --- Helper Functions ---
def draw_board(image, board):
    """Draws the tic-tac-toe board and symbols onto the camera feed."""
    h, w, _ = image.shape
    offset_x = (w - (3 * CELL_SIZE)) // 2
    offset_y = (h - (3 * CELL_SIZE)) // 2

    for i in range(1, 3):
        cv2.line(image, (offset_x + i*CELL_SIZE, offset_y), (offset_x + i*CELL_SIZE, offset_y + 3*CELL_SIZE), BOARD_COLOR, 3)
        cv2.line(image, (offset_x, offset_y + i*CELL_SIZE), (offset_x + 3*CELL_SIZE, offset_y + i*CELL_SIZE), BOARD_COLOR, 3)

    for i, symbol in enumerate(board):
        if symbol:
            row, col = divmod(i, 3)
            center_x = offset_x + col * CELL_SIZE + CELL_SIZE // 2
            center_y = offset_y + row * CELL_SIZE + CELL_SIZE // 2
            color = SYMBOL_COLOR_X if symbol == 'X' else SYMBOL_COLOR_O
            cv2.putText(image, symbol, (center_x - 40, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 10)

def get_selected_cell(finger_pos, image_shape):
    """Determines which cell (0-8) the finger is pointing at."""
    h, w = image_shape
    offset_x = (w - (3 * CELL_SIZE)) // 2
    offset_y = (h - (3 * CELL_SIZE)) // 2
    
    if offset_x < finger_pos[0] < offset_x + 3 * CELL_SIZE and offset_y < finger_pos[1] < offset_y + 3 * CELL_SIZE:
        col = (finger_pos[0] - offset_x) // CELL_SIZE
        row = (finger_pos[1] - offset_y) // CELL_SIZE
        return row * 3 + col
    return None

def check_win(board):
    win_conditions = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for wc in win_conditions:
        if board[wc[0]] == board[wc[1]] == board[wc[2]] and board[wc[0]] != '':
            return board[wc[0]]
    if '' not in board: return 'draw'
    return ''

def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]; pips = [3, 6, 10, 14, 18]; res = []
    # Thumb (for a right hand)
    res.append(hand.landmark[tips[0]].x < hand.landmark[pips[0]].x)
    # Other four fingers
    for i in range(1, 5): res.append(hand.landmark[tips[i]].y < hand.landmark[pips[i]].y)
    return res

def crash_computer():
    """Opens 100 tabs to a chaotic website."""
    print("YOU LOSE! PREPARE FOR CHAOS.")
    url = "https://www.omfgdogs.com/"
    for _ in range(100):
        webbrowser.open(url)

# Firebase listener
def on_snapshot(doc_snapshot, changes, read_time):
    global game_state
    with state_lock:
        if doc_snapshot:
            game_state = doc_snapshot[0].to_dict()

# --- Main Program ---
if __name__ == "__main__":
    choice = input("Do you want to (c)reate or (j)oin a game? ").lower()

    if choice == 'c':
        MY_SYMBOL = 'X'
        GAME_ID = str(random.randint(1000, 9999))
        print(f"Game created! Your Game ID is: {GAME_ID}")
        print("Share this ID with your friend and wait for them to join.")
        game_ref = db.collection('tictactoe_games').document(GAME_ID)
        game_ref.set({
            'board': [''] * 9, 'turn': 'X', 'winner': '',
            'players': {'X': True, 'O': False}
        })
    elif choice == 'j':
        MY_SYMBOL = 'O'
        GAME_ID = input("Enter the 4-digit Game ID: ")
        game_ref = db.collection('tictactoe_games').document(GAME_ID)
        try:
            game_ref.update({'players.O': True})
            print(f"Successfully joined game {GAME_ID}. You are Player O.")
        except Exception as e:
            print(f"Could not join game. Is the Game ID correct? Error: {e}")
            exit()
    else:
        print("Invalid choice.")
        exit()

    game_stream = game_ref.on_snapshot(on_snapshot)

    while True:
        with state_lock:
            local_game_state = game_state.copy()
        
        if not local_game_state or not local_game_state.get('players', {}).get('O'):
            print("Waiting for Player O to join...")
            time.sleep(2)
            continue

        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # --- Gesture detection and move making (only on your turn) ---
        if local_game_state.get('turn') == MY_SYMBOL and not local_game_state.get('winner'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                num_fingers = sum(fingers_up(hand_landmarks))
                gesture_made = (MY_SYMBOL == 'X' and num_fingers == 5) or \
                               (MY_SYMBOL == 'O' and num_fingers == 0)

                if gesture_made and time.time() - last_move_time > 2:
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    pointer_pos = (int(wrist.x * w), int(wrist.y * h))
                    cell = get_selected_cell(pointer_pos, (h, w))

                    if cell is not None and local_game_state['board'][cell] == '':
                        print(f"Player {MY_SYMBOL} places mark in cell {cell}")
                        last_move_time = time.time()
                        new_board = local_game_state['board'][:]
                        new_board[cell] = MY_SYMBOL
                        winner = check_win(new_board)
                        game_ref.update({
                            'board': new_board,
                            'turn': 'O' if MY_SYMBOL == 'X' else 'X',
                            'winner': winner
                        })
        
        # --- Drawing and Display ---
        draw_board(frame, local_game_state.get('board', [''] * 9))
        
        status_text = ""
        winner = local_game_state.get('winner')
        if winner:
            status_text = f"Winner: {winner}!" if winner != 'draw' else "It's a Draw!"
        else:
            turn = local_game_state.get('turn')
            status_text = "Your Turn" if turn == MY_SYMBOL else f"Waiting for {turn}..."
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        cv2.imshow(f"Tic-Tac-Toe - Player {MY_SYMBOL}", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        
        if local_game_state.get('winner'):
            print(f"Game over! Winner is {local_game_state.get('winner')}")
            time.sleep(3)
            break

    # --- Cleanup ---
    game_stream.unsubscribe()
    cap.release()
    cv2.destroyAllWindows()
    
    # --- The Punishment ---
    final_winner = local_game_state.get('winner')
    if final_winner and final_winner != 'draw' and final_winner != MY_SYMBOL:
        crash_computer()

