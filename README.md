Tic-Tac-Crash 💥
A two-player, gesture-controlled Tic-Tac-Toe game with a chaotic twist for the loser.

This project allows two players to play Tic-Tac-Toe in real-time using their webcams for input. Instead of clicking, players use hand gestures to place their marks on the board. The game state is synced instantly between players over the internet using a central Firebase server. And to raise the stakes, the loser's computer is "punished" by having 100 browser tabs opened automatically.

![Gif of the Tic-Tac-Crash game in action]

Features
Real-Time Multiplayer: Play against a friend on a different computer, with moves appearing instantly.

Gesture Controls: No mouse or keyboard required!

Player X uses an Open Palm to make a move.

Player O uses a Fist to make a move.

IP-less Matchmaking: Uses a simple 4-digit Game ID to connect, so you never have to share confusing IP addresses.

The Punishment: The loser's script automatically opens 100 tabs of a chaotic website.

🛠️ Tech Stack
Python 3

OpenCV: For capturing the webcam feed and drawing the game board.

MediaPipe: For real-time hand tracking and gesture recognition.

Firebase Firestore: Acts as the central server for real-time database syncing and matchmaking.

🚀 Setup and Installation (5-10 minutes)
Follow these steps to get the game running. Both players need to set up their environment.

Step 1: Prerequisites
Make sure you have Python 3.8 - 3.11 installed on your system. Anything older/newer than this will NOT work.

Step 2: Get the Code
Clone or download the project files to a folder on your computer. You will need the ttt_2player.py and requirements.txt files.

Step 3: Set up a Virtual Environment
It is crucial to use a virtual environment to avoid conflicts with other Python projects.

Open a terminal or command prompt in your project folder.

Create the environment:

python -m venv env

Activate it:

Windows: .\env\Scripts\activate

macOS/Linux: source env/bin/activate

You should now see (env) at the beginning of your terminal prompt.

Step 4: Install Dependencies
With your virtual environment active, run the following command. This will read the requirements.txt file and install all the necessary libraries automatically.

pip install -r requirements.txt
