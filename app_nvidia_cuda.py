from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import sqlite3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import Delegate
import time
import os
import urllib.request
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from ultralytics import YOLO
from chit_classifier import ChitClassifier
import torch

app = Flask(__name__)

# --- GPU Check for NVIDIA/CUDA ---
def check_cuda():
    print("-" * 30)
    if torch.cuda.is_available():
        print(f"✅ NVIDIA GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return "cuda"
    else:
        print("❌ CUDA not found. Falling back to CPU.")
        print("Hint: Install CUDA-enabled torch with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return "cpu"

device = check_cuda()

# --- 1. Auto-Download Model ---
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

hand_model_path = 'hand_landmarker.task'
if not os.path.exists(hand_model_path):
    print("Downloading hand_landmarker.task model...")
    hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(hand_url, hand_model_path)

# --- 2. Initialize MediaPipe with GPU Delegate ---
# Note: MediaPipe GPU support on Windows Python can be hit-or-miss; we fallback to CPU if initialization fails.
try:
    print("Initializing MediaPipe with GPU acceleration...")
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=Delegate.GPU)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=15,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    base_options_hands = python.BaseOptions(model_asset_path=hand_model_path, delegate=Delegate.GPU)
    options_hands = vision.HandLandmarkerOptions(
        base_options=base_options_hands,
        num_hands=15
    )
    hands_detector_mp = vision.HandLandmarker.create_from_options(options_hands)
    print("✅ MediaPipe GPU initialization successful.")
except Exception as e:
    print(f"⚠️ MediaPipe GPU failed ({e}). Falling back to CPU...")
    # Fallback to CPU
    base_options = python.BaseOptions(model_asset_path=model_path, delegate=Delegate.CPU)
    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_options, num_faces=15)
    )
    base_options_hands = python.BaseOptions(model_asset_path=hand_model_path, delegate=Delegate.CPU)
    hands_detector_mp = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base_options_hands, num_hands=15)
    )

# --- 2.5 Initialize YOLOv8 with CUDA ---
try:
    # Load model and move to CUDA device immediately
    yolo_model = YOLO('yolov8s.pt')
    yolo_model.to(device)
    print(f'✅ YOLOv8s model loaded on {device}.')
except Exception as e:
    print('Failed to load YOLOv8s model:', e)
    yolo_model = None

# stricter device list
cheating_objects = {'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'mobile phone'}

chit_detector = ChitClassifier(min_area=300, max_area=4000)

def is_near(bbox1, bbox2, iou_thresh=0.05):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0: return False
    return (interArea / unionArea) >= iou_thresh

class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005) # Faster update for GPU version

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# Global variables
current_alert = "System (CUDA) Initialized. Monitoring active."
is_critical = False
direction_history = []
last_log_time = {}

def init_db():
    conn = sqlite3.connect('cheatcam.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS incidents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  location TEXT,
                  student_id TEXT,
                  infraction TEXT,
                  status TEXT)''')
    conn.commit()
    conn.close()

def log_incident(student_id, infraction, status):
    current_time = time.time()
    key = f"{student_id}_{infraction}"
    if key in last_log_time and current_time - last_log_time[key] < 5:
        return
    last_log_time[key] = current_time
    conn = sqlite3.connect('cheatcam.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO incidents (timestamp, location, student_id, infraction, status) VALUES (?, ?, ?, ?, ?)",
              (timestamp, "CS-101 (GPU)", student_id, infraction, status))
    conn.commit()
    conn.close()

def detect_head_direction(landmarks, frame_w, frame_h):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_x, left_eye_x, right_eye_x = int(nose.x * frame_w), int(left_eye.x * frame_w), int(right_eye.x * frame_w)
    eye_center = (left_eye_x + right_eye_x) / 2
    threshold = abs(right_eye_x - left_eye_x) * 0.25 
    if nose_x < eye_center - threshold: return "Looking Left"
    elif nose_x > eye_center + threshold: return "Looking Right"
    else: return "Looking Center"

def generate_frames():
    global current_alert, is_critical
    video_url = "http://192.168.1.36:8080/video" 
    # Use thread to avoid buffering lag with IP Camera
    cap_thread = VideoCaptureThread(video_url).start()

    try:
        while True:
            ret, frame = cap_thread.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 1. MediaPipe Inference (GPU Delegate)
            results = detector.detect(mp_image)
            hands_results = hands_detector_mp.detect(mp_image)
            
            frame_h, frame_w, _ = frame.shape
            current_alert = "Monitoring (CUDA Accelerated)..."
            is_critical = False

            # 2. YOLOv8 Inference (CUDA Device)
            yolo_candidates = []
            mobile_count = 0
            if yolo_model is not None:
                # device parameter ensures CUDA utilization if available
                yolo_results = yolo_model(rgb_frame, imgsz=640, conf=0.35, device=device, verbose=False)
                if len(yolo_results) > 0:
                    yolo_res = yolo_results[0]
                    for box in yolo_res.boxes:
                        cls_id = int(box.cls.cpu().numpy()[0])
                        name = yolo_res.names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf.cpu().numpy()[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

                        if name in cheating_objects and conf >= 0.25:
                            yolo_candidates.append((x1, y1, x2, y2, name, conf))
                            mobile_count += 1

            # 3. Hand Detection Logic
            hand_boxes = []
            if hands_results.hand_landmarks:
                for hand_landmarks in hands_results.hand_landmarks:
                    x_coords = [int(lm.x * frame_w) for lm in hand_landmarks]
                    y_coords = [int(lm.y * frame_h) for lm in hand_landmarks]
                    hx_min, hx_max = max(0, min(x_coords) - 60), min(frame_w, max(x_coords) + 60)
                    hy_min, hy_max = max(0, min(y_coords) - 60), min(frame_h, max(y_coords) + 60)
                    hand_boxes.append((hx_min, hy_min, hx_max, hy_max))
                    cv2.rectangle(frame, (hx_min, hy_min), (hx_max, hy_max), (255, 0, 255), 1)

            # 4. Chit Classifier (CPU, but lightweight)
            raw_chits = chit_detector.detect(frame)
            chit_candidates = []
            for cx1, cy1, cx2, cy2, cname, cconf in raw_chits:
                near_hand = any(max(0, min(cx2, hx2) - max(cx1, hx1)) > 0 and 
                                max(0, min(cy2, hy2) - max(cy1, hy1)) > 0 for (hx1, hy1, hx2, hy2) in hand_boxes)
                if near_hand:
                    chit_candidates.append((cx1, cy1, cx2, cy2, cname, cconf))

            chit_count = len(chit_candidates)
            for cx1, cy1, cx2, cy2, cname, cconf in chit_candidates:
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
                cv2.putText(frame, f"{cname}", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            # 5. Face Tracking & Head Pose
            if results.face_landmarks:
                total_faces = len(results.face_landmarks)
                cv2.putText(frame, f"Students (GPU): {total_faces}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                suspect_count = 0
                face_boxes = []

                for face_id, face_landmarks in enumerate(results.face_landmarks):
                    direction = detect_head_direction(face_landmarks, frame_w, frame_h)
                    direction_history.append((time.time(), direction))
                    x_coords = [int(lm.x * frame_w) for lm in face_landmarks]
                    y_coords = [int(lm.y * frame_h) for lm in face_landmarks]
                    x_min, y_min = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
                    x_max, y_max = min(frame_w, max(x_coords) + 20), min(frame_h, max(y_coords) + 20)
                    face_boxes.append((x_min, y_min, x_max, y_max))

                    box_color = (0, 255, 0)
                    if direction != "Looking Center":
                        suspect_count += 1
                        box_color = (0, 0, 255)
                        log_incident(f"ID:{face_id}", direction, "Flagged")
                        cv2.putText(frame, f"ID:{face_id} {direction}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"ID:{face_id} Clear", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    for lm_index in [1, 33, 263]:
                        lm = face_landmarks[lm_index]
                        cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 3, box_color, -1)

                # Look around detection
                current_time = time.time()
                direction_history[:] = [h for h in direction_history if current_time - h[0] <= 30]
                change_count = 0
                prev_dir = None
                for _, dir in sorted(direction_history, key=lambda x: x[0]):
                    if prev_dir is not None and dir != prev_dir: change_count += 1
                    prev_dir = dir
                looking_around = change_count >= 5

                # Proximity checks
                for x1, y1, x2, y2, name, conf in yolo_candidates:
                    if any(is_near(fb, (x1, y1, x2, y2)) for fb in face_boxes):
                        mobile_count += 1
                        log_incident("Unknown", f"{name} near face", "Critical")
                        cv2.putText(frame, f"{name} ALERT", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                for cx1, cy1, cx2, cy2, cname, cconf in chit_candidates:
                    if any(is_near(fb, (cx1, cy1, cx2, cy2)) for fb in face_boxes):
                        chit_count += 1
                        log_incident("Unknown", f"{cname} near face", "Critical")
                        cv2.putText(frame, f"{cname} ALERT", (cx1, cy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                # Alerting Logic
                total_devices = mobile_count + chit_count
                if suspect_count > 0 and total_devices > 0:
                    current_alert = f"CRITICAL: {suspect_count} suspicious & {total_devices} device(s) [CUDA]"
                    is_critical = True
                elif suspect_count > 0 or total_devices > 0 or looking_around:
                    current_alert = f"WARNING: Activity Detected [{ 'Head' if suspect_count else '' }{ ' Device' if total_devices else '' }{ ' Movement' if looking_around else '' }]".strip()
                    is_critical = True
                    if looking_around: log_incident("Room", "Frequent movement", "Flagged")
                else:
                    current_alert = "Room clear (CUDA Monitoring)."
                    is_critical = False
            else:
                total_devices = mobile_count + chit_count
                if total_devices > 0:
                    current_alert = f"WARNING: {total_devices} devices detected."
                    is_critical = True
                else:
                    current_alert = "No students detected."
                    is_critical = False
                cv2.putText(frame, "No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap_thread.stop()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/classroom')
def classroom(): return render_template('ClassRoom.html')

@app.route('/database')
def database():
    conn = sqlite3.connect('cheatcam.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, location, student_id, infraction, status FROM incidents ORDER BY id DESC")
    incidents = c.fetchall()
    conn.close()
    return render_template('Database.html', incidents=incidents)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alert')
def get_alert(): return jsonify({'alert': current_alert, 'critical': is_critical})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000, threaded=True)
