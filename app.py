from flask import Flask, render_template, Response, jsonify
import cv2
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

app = Flask(__name__)

# --- 1. Auto-Download Model ---
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# --- 2. Initialize MediaPipe ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=15, # Track up to 15 students at once
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 2.5 Initialize YOLOv8 for object/mobile detection ---
try:
    yolo_model = YOLO('yolov8n.pt')
    print('YOLOv8n model loaded successfully.')
except Exception as e:
    print('Failed to load YOLOv8n model:', e)
    yolo_model = None

# stricter device list, avoids bottle/furniture false positive as phone
cheating_objects = {'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'mobile phone'}

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

# Global variables to pass data to the frontend
current_alert = "System Initialized. Monitoring active."
is_critical = False
direction_history = []  # list of (timestamp, direction)

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
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

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

                        # draw all detections as reference
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

                        # candidate filtering is stricter to avoid bottle and big objects
                        if name in cheating_objects and conf >= 0.25:
                            yolo_candidates.append((x1, y1, x2, y2, name, conf))
                            mobile_count += 1  # Count all cheating devices in frame
            except Exception as e:
                print('YOLO detection error:', e)

        if results.face_landmarks:
            total_faces = len(results.face_landmarks)
            cv2.putText(frame, f"Students Tracking: {total_faces}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            suspect_count = 0
            face_boxes = []

            # Loop through EVERY face detected in the classroom
            for face_id, face_landmarks in enumerate(results.face_landmarks):
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
                        cv2.putText(frame, f"{name} detected near face", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                        break

            # Update the web dashboard alerts based on the whole room and object detection
            if suspect_count > 0 and mobile_count > 0:
                current_alert = f"CRITICAL: {suspect_count} suspicious student(s) and {mobile_count} potential cheating device(s) detected!"
                is_critical = True
            elif suspect_count > 0:
                current_alert = f"WARNING: {suspect_count} student(s) showing suspicious behavior!"
                is_critical = True
            elif mobile_count > 0:
                current_alert = f"WARNING: {mobile_count} potential cheating device(s) detected!"
                is_critical = True
            elif looking_around:
                current_alert = "WARNING: Frequent head movement detected - possible looking around."
                is_critical = True
            else:
                if not is_critical:
                    current_alert = "Room clear. All students focused."
                    is_critical = False

        else:
            # If no faces are found at all
            if mobile_count > 0:
                current_alert = f"WARNING: {mobile_count} potential cheating device(s) detected!"
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
    # Links to "Alert Database"
    return render_template('Database.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alert')
def get_alert():
    return jsonify({'alert': current_alert, 'critical': is_critical})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    