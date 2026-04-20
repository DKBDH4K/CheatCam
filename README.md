# AI CheatCam - AI Based Cheating Detection System

CheatCam is an advanced, AI-powered multi-modal classroom monitoring system designed to detect anomalous behavior and potential academic cheating in real-time. Built using Flask, OpenCV, MediaPipe, and YOLOv8, CheatCam processes live video feeds to track student head movements, identify prohibited objects, and detect hidden cheat sheets (chits). 

By combining object detection with facial landmarking through a custom two-stage pipeline, CheatCam performs reliably even on students seated far away from the camera.

## 🚀 Key Features

*   **Two-Stage Long-Distance Head Tracking:** Employs YOLOv8 to locate students dynamically, crops their upper body, and feeds the high-resolution crop to MediaPipe's Face Landmarker. This ensures highly accurate "Looking Left / Right / Center" gaze detection even at maximum classroom depth.
*   **Prohibited Object Detection (YOLOv8):** Continuously scans for unauthorized devices like mobile phones, laptops, and smart remotes.
*   **Custom Paper "Chit" Classifier:** Utilizes computer vision and HSV color-space thresholding to find small pieces of paper (cheat sheets).
*   **Proximity-Based Validation:** Uses MediaPipe Hand Tracking to confirm that detected chits or objects are actually near a student's hand, reducing false positives from background clutter.
*   **Automated Incident Logging:** Suspicious actions (e.g., frequent looking around, devices near face) are automatically logged with a timestamp into a local SQLite database (`cheatcam.db`).
*   **Real-Time Web Dashboard:** A responsive Flask-based web interface streams the live annotated video feed and provides unified room alerts (Safe, Warning, Critical) and incident history logs.

## 🛠️ Technology Stack

*   **Backend:** Python 3, Flask, SQLite3
*   **Computer Vision:** OpenCV (`cv2`)
*   **Machine Learning Models:**
    *   **Ultralytics YOLOv8** (`yolov8s.pt`): For person and object detection.
    *   **Google MediaPipe Tasks API**: For Face Landmarking (`face_landmarker.task`) and Hand Landmarking (`hand_landmarker.task`).
*   **Frontend:** HTML5, CSS3, JavaScript (Jinja2 Templates)

## 📁 Project Structure

```text
CheatCam/
├── app.py                 # Core Flask application and CV pipeline
├── chit_classifier.py     # Custom OpenCV module to detect small white paper chits
├── requirements.txt       # Python dependency list
├── cheatcam.db            # SQLite database for storing incident logs
├── yolov8s.pt             # YOLOv8 Small model weights
├── face_landmarker.task   # MediaPipe Face weights (Auto-downloaded if missing)
├── hand_landmarker.task   # MediaPipe Hand weights (Auto-downloaded if missing)
├── static/                # CSS/JS and static assets
└── templates/             # HTML templates 
    ├── index.html         # Landing page
    ├── dashboard.html     # Alert dashboard
    ├── classroom.html     # Live camera feed view
    └── Database.html      # Incident review table
```

## ⚙️ Setup and Installation

1.  **Clone the Repository** and navigate to the project root.
2.  **Install Dependencies:**
    Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```
    *Required packages generally include: `flask`, `opencv-python`, `mediapipe`, `ultralytics`, `numpy`, and `firebase-admin` (if using cloud integration).*
3.  **Model Downloads:** 
    The YOLOv8 model (`yolov8s.pt`) and MediaPipe tasks are required. `app.py` is configured to auto-download the MediaPipe `.task` files on the first run if they are missing.

## ▶️ Running the Application

1.  Start the Flask backend:
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to:
    ```
    http://localhost:5000/
    ```
3.  *Note on Camera Source:* By default, `app.py` is configured to hook into an IP camera url (`http://172.30.233.205:8080/video`). To use a standard local webcam, you can edit line 166 in `app.py` to `video_url = 0`.

## 🧠 How it Works

1.  **Frame Capture:** Video frames are grabbed using a dedicated asynchronous thread (`VideoCaptureThread`) to prevent IP-camera buffering lag.
2.  **YOLO Sweep:** The entire frame is searched for `person` and `cheating_objects` (phones).
3.  **Hybrid Zoom:** For every `person` found, a zoomed-in crop is sent to MediaPipe Face Landmarker to compute the rotation of facial landmarks (Points 1, 33, 263).
4.  **Hand & Chit Check:** MediaPipe maps all hands. The `ChitClassifier` maps all papers. If a paper bounding box spatially overlaps with a hand bounding box, an alert is triggered.
5.  **Heuristics:** 5+ head direction changes within a 30-second window triggers a "looking around" warning.
