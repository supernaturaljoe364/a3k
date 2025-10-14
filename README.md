Tic-Tac-Crash 💥
A two-player, gesture-controlled Tic-Tac-Toe game with a chaotic twist for the loser.

This project allows two players to play Tic-Tac-Toe in real-time using their webcams for input. Instead of clicking, players use hand gestures to place their marks. The game state is synced instantly between players over the internet. And to raise the stakes, the loser's computer is "punished" by having 100 browser tabs opened automatically.

Features
Real-Time Multiplayer: Play against a friend on a different computer, with moves appearing instantly.

Gesture Controls: No mouse or keyboard required!

Player X uses an Open Palm to make a move.

Player O uses a Fist to make a move.

IP-less Matchmaking: Uses a simple 4-digit Game ID to connect, so you never have to share confusing IP addresses.

The Punishment: The loser's script automatically opens 100 tabs of a chaotic website.

Tech Stack
Python 3

OpenCV: For capturing the webcam feed and drawing the game board.

MediaPipe: For real-time hand tracking and gesture recognition.

Firebase Firestore: Acts as the central server for real-time database syncing and matchmaking, eliminating the need for direct P2P connections.

🚀 Setup and Installation (5-10 minutes)
Follow these steps to get the game running. Both players need to do this.

Step 1: Prerequisites
Make sure you have Python 3.8 or newer installed on your system.

Step 2: Get the Code
Download or clone the project files to a folder on your computer. You should have:

tic_tac_toe_firebase.py

README.md (this file)

Step 3: Set up a Virtual Environment
It's highly recommended to use a virtual environment to keep project dependencies separate.

Open a terminal or command prompt in your project folder.

Create the environment:

python -m venv env

Activate it:

On Windows: .\env\Scripts\activate

On macOS/Linux: source env/bin/activate

You should see (env) at the beginning of your terminal prompt.

Step 4: Install Python Libraries
With your virtual environment active, run the following command to install all the necessary libraries:

pip install opencv-python mediapipe firebase-admin

Troubleshooting: If you see a protobuf version error after installation, run this command to fix it:

pip install protobuf==4.25.3

Step 5: Set up the Firebase Project (Only one player needs to do this)
The game needs a central Firebase project to act as the server. This is free.

Create a Firebase Project:

Go to the Firebase Console.

Click "Add project", give it a name (e.g., "TicTacCrash"), and create it. You can disable Google Analytics.

Create a Firestore Database:

In your project, click "Firestore Database" from the left menu.

Click "Create database".

Select "Start in test mode" (this allows the script to connect easily).

Choose a server location and click Enable.

Get Your Credentials:

Click the gear icon ⚙️ next to "Project Overview" and go to Project settings.

Go to the Service accounts tab.

Click "Generate new private key".

A JSON file will be downloaded. Rename this file to credentials.json and place it in the same folder as your tic_tac_toe_firebase.py script.

Share this credentials.json file with your friend. Both of you need it in your project folders.

▶️ How to Play
Player 1 (Create a Game):

Open your terminal in the project folder (with the virtual environment active).

Run the script: python tic_tac_toe_firebase.py

When prompted, type c and press Enter.

A 4-digit Game ID will be displayed. Share this ID with Player 2.

Player 2 (Join a Game):

Open your terminal in the project folder (with the virtual environment active).

Run the script: python tic_tac_toe_firebase.py

When prompted, type j and press Enter.

Enter the 4-digit Game ID you received from Player 1.

Play!

The game window will open for both players.

Player X (the creator) uses an Open Palm to place their mark.

Player O (the joiner) uses a Fist to place their mark.

The position of your wrist determines which cell you are aiming at.

Good luck, and try not to lose!
