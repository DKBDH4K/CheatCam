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
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from ultralytics import YOLO
from chit_classifier import ChitClassifier

app = Flask(__name__)

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

# --- 2. Initialize MediaPipe ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=15, # Track up to 15 students at once
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_face_detection_confidence=0.1,
    min_face_presence_confidence=0.1,
    min_tracking_confidence=0.1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- MediaPipe Hands Initialization (Modern Tasks API) ---
base_options_hands = python.BaseOptions(model_asset_path=hand_model_path)
options_hands = vision.HandLandmarkerOptions(
    base_options=base_options_hands,
    num_hands=15
)
hands_detector_mp = vision.HandLandmarker.create_from_options(options_hands)

# --- 2.5 Initialize YOLOv8 for object/mobile detection ---
try:
    yolo_model = YOLO('yolov8s.pt')
    print('YOLOv8s model loaded successfully.')
except Exception as e:
    print('Failed to load YOLOv8s model:', e)
    yolo_model = None

# stricter device list, avoids bottle/furniture false positive as phone
cheating_objects = {'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'mobile phone'}

# Initialize Chit Classifier for small foreign paper detection
chit_detector = ChitClassifier(min_area=300, max_area=4000)

# helper for proximity check

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
    if unionArea <= 0:
        return False
    iou = interArea / unionArea
    return iou >= iou_thresh

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
            time.sleep(0.01)  # small sleep to avoid cpu hogging

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Global variables to pass data to the frontend
current_alert = "System Initialized. Monitoring active."
is_critical = False
direction_history = []  # list of (timestamp, direction)
last_log_time = {}      # dict of student_id: last_logged_timestamp

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
    # 5-second cooldown per student/infraction to avoid spam
    key = f"{student_id}_{infraction}"
    if key in last_log_time and current_time - last_log_time[key] < 5:
        return
    
    last_log_time[key] = current_time
    
    conn = sqlite3.connect('cheatcam.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO incidents (timestamp, location, student_id, infraction, status) VALUES (?, ?, ?, ?, ?)",
              (timestamp, "CS-101", student_id, infraction, status))
    conn.commit()
    conn.close()

def detect_head_direction(landmarks, frame_w, frame_h):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_x, left_eye_x, right_eye_x = int(nose.x * frame_w), int(left_eye.x * frame_w), int(right_eye.x * frame_w)
    eye_center = (left_eye_x + right_eye_x) / 2
    threshold = abs(right_eye_x - left_eye_x) * 0.15 

    if nose_x < eye_center - threshold: return "Looking Left"
    elif nose_x > eye_center + threshold: return "Looking Right"
    else: return "Looking Center"

def generate_frames():
    global current_alert, is_critical
    video_url = "http://172.30.233.205:8080/video" 
    # Use thread to avoid buffering lag with IP Camera
    cap_thread = VideoCaptureThread(video_url).start()

    try:
        while True:
            ret, frame = cap_thread.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            frame_h, frame_w, _ = frame.shape

            # Reset alert state for the current frame
            current_alert = "Monitoring..."
            is_critical = False

            # YOLOv8 object/mobile detection (cheating device detection)
            yolo_candidates = []
            person_boxes = []
            mobile_count = 0
            if yolo_model is not None:
                try:
                    yolo_results = yolo_model(rgb_frame, imgsz=640, conf=0.35, device='cpu')
                    if len(yolo_results) > 0:
                        yolo_res = yolo_results[0]
                        for box in yolo_res.boxes:
                            cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else int(box.cls)
                            name = yolo_res.names.get(cls_id, str(cls_id)) if hasattr(yolo_res, 'names') else str(cls_id)
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()) if hasattr(box, 'xyxy') else map(int, box.xyxy[0])
                            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else float(box.conf)

                            if name == 'person':
                                person_boxes.append((x1, y1, x2, y2))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
                            elif name in cheating_objects and conf >= 0.25:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
                                yolo_candidates.append((x1, y1, x2, y2, name, conf))
                                mobile_count += 1  # Count all cheating devices in frame
                except Exception as e:
                    print('YOLO detection error:', e)

            # --- Hand Detection (for Proximity Verification) ---
            hands_results = hands_detector_mp.detect(mp_image)
            hand_boxes = []
            if hands_results.hand_landmarks:
                for hand_landmarks in hands_results.hand_landmarks:
                    x_coords = [int(lm.x * frame_w) for lm in hand_landmarks]
                    y_coords = [int(lm.y * frame_h) for lm in hand_landmarks]
                    
                    # Create a generous pad around the hand to catch papers held or nearby
                    hx_min, hx_max = max(0, min(x_coords) - 60), min(frame_w, max(x_coords) + 60)
                    hy_min, hy_max = max(0, min(y_coords) - 60), min(frame_h, max(y_coords) + 60)
                    hand_boxes.append((hx_min, hy_min, hx_max, hy_max))
                    
                    # Draw a magenta boundary box around detected hands for visual clarity
                    cv2.rectangle(frame, (hx_min, hy_min), (hx_max, hy_max), (255, 0, 255), 1)

            # --- Chit Detection Integration ---
            raw_chits = chit_detector.detect(frame)
            chit_candidates = []
            
            # PROXIMITY CHECK: Only accept chits if their bounding box overlaps with a Hand Zone
            for cx1, cy1, cx2, cy2, cname, cconf in raw_chits:
                near_hand = False
                for (hx1, hy1, hx2, hy2) in hand_boxes:
                    # Check for boundary intersection instead of strict IOU
                    interW = max(0, min(cx2, hx2) - max(cx1, hx1))
                    interH = max(0, min(cy2, hy2) - max(cy1, hy1))
                    if interW > 0 and interH > 0:
                        near_hand = True
                        break
                        
                if near_hand:
                    chit_candidates.append((cx1, cy1, cx2, cy2, cname, cconf))

            chit_count = len(chit_candidates)
            for cx1, cy1, cx2, cy2, cname, cconf in chit_candidates:
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2) # Orange bounding box for chit
                cv2.putText(frame, f"{cname}", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            # --- Combine Full Frame and Cropped Faces ---
            all_face_landmarks = list(results.face_landmarks) if results.face_landmarks else []
            
            for px1, py1, px2, py2 in person_boxes:
                # Capture the full person box with generous margins
                cx1, cy1 = max(0, px1 - 50), max(0, py1 - 50)
                cx2, cy2 = min(frame_w, px2 + 50), min(frame_h, py2 + 50)
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                
                if crop_w > 20 and crop_h > 20:
                    crop_rgb = rgb_frame[cy1:cy2, cx1:cx2].copy()
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                    crop_results = detector.detect(mp_crop)
                    
                    if crop_results.face_landmarks:
                        for crop_face in crop_results.face_landmarks:
                            global_face = []
                            for lm in crop_face:
                                global_x = (lm.x * crop_w + cx1) / frame_w
                                global_y = (lm.y * crop_h + cy1) / frame_h
                                global_face.append(MockLandmark(global_x, global_y))
                            
                            # Check for duplicates using Nose position (index 1)
                            is_duplicate = False
                            new_nose = global_face[1]
                            for existing_face in all_face_landmarks:
                                ex_nose = existing_face[1]
                                dist = ((new_nose.x - ex_nose.x) * frame_w)**2 + ((new_nose.y - ex_nose.y) * frame_h)**2
                                if dist < 2500: # 50 pixels distance threshold for same face
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                all_face_landmarks.append(global_face)

            if all_face_landmarks:
                total_faces = len(all_face_landmarks)
                cv2.putText(frame, f"Students Tracking: {total_faces}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                suspect_count = 0
                face_boxes = []

                # Loop through EVERY face detected in the classroom
                for face_id, face_landmarks in enumerate(all_face_landmarks):
                    direction = detect_head_direction(face_landmarks, frame_w, frame_h)
                    direction_history.append((time.time(), direction))
                    
                    # Calculate Bounding Box coordinates
                    x_coords = [int(lm.x * frame_w) for lm in face_landmarks]
                    y_coords = [int(lm.y * frame_h) for lm in face_landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Expand the bounding box slightly around the face
                    x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)
                    x_max, y_max = min(frame_w, x_max + 20), min(frame_h, y_max + 20)
                    face_boxes.append((x_min, y_min, x_max, y_max))

                    # Determine box color based on cheating status
                    box_color = (0, 255, 0) # Green for looking center
                    
                    if direction != "Looking Center":
                        suspect_count += 1
                        box_color = (0, 0, 255) # Red for cheating
                        # Log to database
                        log_incident(f"ID:{face_id}", direction, "Flagged")
                        # Draw a warning directly over the cheating student's head
                        cv2.putText(frame, f"ID:{face_id} {direction}", (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Draw safe status
                        cv2.putText(frame, f"ID:{face_id} Clear", (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw the Bounding Box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

                    # Draw tracking dots (Nose, Left Eye, Right Eye)
                    for lm_index in [1, 33, 263]:
                        lm = face_landmarks[lm_index]
                        cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 3, box_color, -1)

                # Process direction history for looking around detection
                current_time = time.time()
                # Keep only last 30 seconds
                direction_history[:] = [h for h in direction_history if current_time - h[0] <= 30]
                # Sort by time
                direction_history.sort(key=lambda x: x[0])
                # Count direction changes
                change_count = 0
                prev_dir = None
                for _, dir in direction_history:
                    if prev_dir is not None and dir != prev_dir:
                        change_count += 1
                    prev_dir = dir
                looking_around = change_count >= 5  # threshold: 5 changes in 30 seconds

                # Count cheating device candidates that are near detected faces
                for x1, y1, x2, y2, name, conf in yolo_candidates:
                    for face_box in face_boxes:
                        if is_near(face_box, (x1, y1, x2, y2), iou_thresh=0.05):
                            mobile_count += 1
                            log_incident(f"Unknown", f"{name} detected near student", "Critical")
                            cv2.putText(frame, f"{name} detected near face", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                            break

                # Count paper chits that are near detected faces
                for cx1, cy1, cx2, cy2, cname, cconf in chit_candidates:
                    for face_box in face_boxes:
                        if is_near(face_box, (cx1, cy1, cx2, cy2), iou_thresh=0.05):
                            chit_count += 1
                            log_incident(f"Unknown", f"{cname} detected near student", "Critical")
                            cv2.putText(frame, f"{cname} near face", (cx1, cy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                            break

                # Update the web dashboard alerts based on the whole room and object detection
                total_devices = mobile_count + chit_count
                if suspect_count > 0 and total_devices > 0:
                    current_alert = f"CRITICAL: {suspect_count} suspicious student(s) and {total_devices} potential cheating material/device(s) detected!"
                    is_critical = True
                elif suspect_count > 0:
                    current_alert = f"WARNING: {suspect_count} student(s) showing suspicious behavior!"
                    is_critical = True
                elif total_devices > 0:
                    current_alert = f"WARNING: {total_devices} potential cheating material/device(s) detected!"
                    is_critical = True
                elif looking_around:
                    current_alert = "WARNING: Frequent head movement detected - possible looking around."
                    is_critical = True
                    log_incident("Room", "Frequent head movement", "Flagged")
                else:
                    if not is_critical:
                        current_alert = "Room clear. All students focused."
                        is_critical = False

            else:
                # If no faces are found at all
                total_devices = mobile_count + chit_count
                if total_devices > 0:
                    current_alert = f"WARNING: {total_devices} potential cheating material/device(s) detected!"
                    is_critical = True
                else:
                    current_alert = "No students detected in frame."
                    is_critical = False
                cv2.putText(frame, "No Face Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap_thread.stop()

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Links to "Room CS-101"
    return render_template('dashboard.html')

@app.route('/classroom')
def classroom():
    # Links to "Room CS-102"
    return render_template('ClassRoom.html')

@app.route('/database')
def database():
    conn = sqlite3.connect('cheatcam.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, location, student_id, infraction, status FROM incidents ORDER BY id DESC")
    incidents = c.fetchall()
    conn.close()
    return render_template('Database.html', incidents=incidents)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alert')
def get_alert():
    return jsonify({'alert': current_alert, 'critical': is_critical})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

    