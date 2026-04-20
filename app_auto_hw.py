from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import sqlite3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request
from datetime import datetime
from ultralytics import YOLO
from chit_classifier import ChitClassifier

app = Flask(__name__)

# --- Hardware Detection ---
def get_best_device():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Hardware Accel: Using GPU ({torch.cuda.get_device_name(0)})")
            return 0
    except: pass
    print("Hardware Accel: Using CPU")
    return 'cpu'

device = get_best_device()

# --- Model Downloads ---
def download_models():
    models = {
        'face_landmarker.task': "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        'hand_landmarker.task': "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    }
    for name, url in models.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, name)

download_models()

# --- Initialize MediaPipe ---
# Defaults to whatever MediaPipe picks as best for the system
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_options, num_faces=15)
)
base_options_hands = python.BaseOptions(model_asset_path='hand_landmarker.task')
hands_detector_mp = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_options_hands, num_hands=15)
)

# --- Initialize YOLO ---
try:
    yolo_model = YOLO('yolov8s.pt')
    print('YOLOv8s loaded.')
except Exception as e:
    print('YOLO load error:', e)
    yolo_model = None

cheating_objects = {'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'mobile phone'}
chit_detector = ChitClassifier(min_area=300, max_area=4000)

class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = False, None
        self.stopped = False
        self.lock = threading.Lock()
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock: self.ret, self.frame = ret, frame
            time.sleep(0.01)
    def read(self):
        with self.lock: return self.ret, self.frame
    def stop(self):
        self.stopped = True
        self.cap.release()

current_alert = "System Initialized (Auto Accel)."
is_critical = False
direction_history = []
last_log_time = {}

def init_db():
    conn = sqlite3.connect('cheatcam.db')
    conn.execute('CREATE TABLE IF NOT EXISTS incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, location TEXT, student_id TEXT, infraction TEXT, status TEXT)')
    conn.close()

def log_incident(student_id, infraction, status):
    key = f"{student_id}_{infraction}"
    if key in last_log_time and time.time() - last_log_time[key] < 5: return
    last_log_time[key] = time.time()
    conn = sqlite3.connect('cheatcam.db')
    conn.execute("INSERT INTO incidents (timestamp, location, student_id, infraction, status) VALUES (?, ?, ?, ?, ?)",
                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "CS-101 (Auto)", student_id, infraction, status))
    conn.commit()
    conn.close()

def detect_head_direction(landmarks, w, h):
    nx, lx, rx = int(landmarks[1].x * w), int(landmarks[33].x * w), int(landmarks[263].x * w)
    center = (lx + rx) / 2
    thresh = abs(rx - lx) * 0.25
    if nx < center - thresh: return "Looking Left"
    if nx > center + thresh: return "Looking Right"
    return "Looking Center"

def is_near(b1, b2):
    return max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) > 0 and max(0, min(b1[3], b2[3]) - max(b1[1], b2[1])) > 0

def generate_frames():
    global current_alert, is_critical
    video_url = "http://192.168.1.36:8080/video" 
    cap = VideoCaptureThread(video_url).start()
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            res_face = detector.detect(mp_img)
            res_hand = hands_detector_mp.detect(mp_img)
            current_alert = "Monitoring (Auto Mode)..."
            is_critical = False

            yolo_cands, m_count = [], 0
            if yolo_model:
                y_res = yolo_model(rgb, imgsz=640, conf=0.35, device=device, verbose=False)
                if y_res:
                    for b in y_res[0].boxes:
                        cls = int(b.cls[0]); name = y_res[0].names[cls]
                        conf = float(b.conf[0]); x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        if name in cheating_objects and conf > 0.25:
                            yolo_cands.append((x1, y1, x2, y2, name))
                            m_count += 1

            h_boxes = []
            if res_hand.hand_landmarks:
                for hl in res_hand.hand_landmarks:
                    xs, ys = [int(lm.x*w) for lm in hl], [int(lm.y*h) for lm in hl]
                    hb = (max(0, min(xs)-60), max(0, min(ys)-60), min(w, max(xs)+60), min(h, max(ys)+60))
                    h_boxes.append(hb)
                    cv2.rectangle(frame, (hb[0], hb[1]), (hb[2], hb[3]), (255, 0, 255), 1)

            raw_chits = chit_detector.detect(frame)
            chits = []
            for c in raw_chits:
                if any(is_near(c[:4], hb) for hb in h_boxes): chits.append(c)

            if res_face.face_landmarks:
                cv2.putText(frame, f"Students: {len(res_face.face_landmarks)}", (30, 40), 1, 1.5, (0, 255, 255), 2)
                s_count, f_boxes = 0, []
                for fid, lm in enumerate(res_face.face_landmarks):
                    dir = detect_head_direction(lm, w, h)
                    direction_history.append((time.time(), dir))
                    xs, ys = [int(l.x*w) for l in lm], [int(l.y*h) for l in lm]
                    fb = (max(0, min(xs)-20), max(0, min(ys)-20), min(w, max(xs)+20), min(h, max(ys)+20))
                    f_boxes.append(fb)
                    col = (0, 255, 0)
                    if dir != "Looking Center":
                        s_count += 1; col = (0, 0, 255)
                        log_incident(f"ID:{fid}", dir, "Flagged")
                        cv2.putText(frame, f"ID:{fid} {dir}", (fb[0], fb[1]-10), 1, 1, col, 2)
                    cv2.rectangle(frame, (fb[0], fb[1]), (fb[2], fb[3]), col, 2)

                # History check
                direction_history[:] = [h for h in direction_history if time.time() - h[0] <= 30]
                changes = sum(1 for i in range(1, len(direction_history)) if direction_history[i][1] != direction_history[i-1][1])
                looking_around = changes >= 5

                for yc in yolo_cands:
                    if any(is_near(fb, yc[:4]) for fb in f_boxes):
                        m_count += 1; log_incident("User", f"{yc[4]} near face", "Critical")

                total = m_count + len(chits)
                if s_count > 0 or total > 0 or looking_around:
                    current_alert = f"WARNING/CRITICAL: Activity Detected!"
                    is_critical = True
                else: current_alert = "Room clear."

            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    finally: cap.stop()

@app.route('/')
def index(): return render_template('index.html')
@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')
@app.route('/database')
def database():
    conn = sqlite3.connect('cheatcam.db')
    incidents = conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()
    conn.close()
    return render_template('Database.html', incidents=incidents)
@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_alert')
def get_alert(): return jsonify({'alert': current_alert, 'critical': is_critical})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
