# Face Puzzle: Gesture-Controlled CV Game 🧩🤳

An interactive computer vision project that transforms a live camera feed into a playable puzzle. This application leverages **MediaPipe** for facial landmarking and hand tracking to create a touchless gaming experience.

## 🚀 How it Works
1. **Face Detection**: The app uses MediaPipe Face Mesh to locate the mouth region.
2. **ROI Capture**: Pressing **'S'** captures the mouth area, resizes it, and slices it into a **3x4 grid**.
3. **Hand Tracking**: MediaPipe Hands tracks your index finger.
4. **Gesture Interaction**: Swiping your hand (Up, Down, Left, or Right) triggers a tile swap.
5. **Win Condition**: The game continuously checks the current tile configuration against the original capture to detect a "Solved" state.

## 🛠️ Tech Stack
* **Python 3.11+**
* **OpenCV**: Camera handling and UI rendering.
* **MediaPipe**: Real-time face and hand landmark detection.
* **NumPy**: Matrix manipulation and tile stacking.

## 🎮 Controls
| Key | Action |
|-----|--------|
| **S** | Capture mouth and start puzzle |
| **R** | Reshuffle tiles |
| **Q** | Quit the game |
| **Hand Swipe** | Swap puzzle pieces |

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/saystalha/FACE-PUZZLE.git](https://github.com/saystalha/FACE-PUZZLE.git)
   cd FACE-PUZZLE
