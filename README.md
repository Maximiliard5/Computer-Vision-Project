# Computer-Vision-Project
This project was developed as part of a university assignment and focuses on using computer vision and image processing techniques to analyze the board game Mathable.

The goal was to automate the detection of new tiles placed on the board, identify the numbers on them, and calculate player scores based on game logic — all from sequential images of the game board.

🔍 Key Features

Board Extraction: Detects and isolates the playing area from fixed-angle photos using OpenCV contour detection and perspective transforms.

Change Detection: Compares consecutive board states to find the position of newly placed tiles.

Number Recognition: Differentiates between single and double-digit numbers using thresholding, contour analysis, and template matching.

Game Logic Integration: Calculates scores according to Mathable rules, including tile multipliers and valid mathematical operations.

Automated Workflow: Processes all input images and outputs text files containing player moves and cumulative scores.

🧠 Technologies Used

Python

OpenCV

NumPy

Matplotlib

os module
