import cv2
import os
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import atexit

# -------------------------------
# ⚙️ Flask Setup
# -------------------------------
app = Flask(__name__)

# Automatically detect working camera index
def find_camera_index():
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            return idx
    return None

camera_index = find_camera_index()
if camera_index is None:
    raise RuntimeError("❌ No camera found. Please connect a webcam.")

cap = cv2.VideoCapture(camera_index)
print(f"✅ Camera opened on index {camera_index}")

# -------------------------------
# 🧠 YOLO + Neural Model
# -------------------------------
print("🔍 Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")

motion_classifier = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
print("⚙️ Motion classifier initialized.")

# -------------------------------
# 📂 Create snapshot folder
# -------------------------------
snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

last_save_time = 0
SAVE_INTERVAL = 3  # seconds between saves
CONF_THRESHOLD = 0.6  # confidence threshold for YOLO detection
MOTION_THRESHOLD = 300000  # motion difference sensitivity


# -------------------------------
# 🎥 Video Feed Generator
# -------------------------------
def generate_frames():
    global last_save_time
    prev_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion detection by frame differencing
        if prev_frame is None:
            prev_frame = gray
            continue

        frame_diff = cv2.absdiff(prev_frame, gray)
        motion_score = np.sum(frame_diff)

        # YOLO detection
        results = yolo_model(frame, verbose=False)
        detected = False

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > CONF_THRESHOLD:
                detected = True
                (x1, y1, x2, y2) = map(int, box.xyxy[0])
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Combine motion + detection for true event
        current_time = time.time()
        if detected and motion_score > MOTION_THRESHOLD and current_time - last_save_time > SAVE_INTERVAL:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(snapshot_dir, f"snap_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"📸 Snapshot saved: {filename}")
            last_save_time = current_time

        prev_frame = gray

        # Encode for web feed
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# -------------------------------
# 🌐 Flask Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manual_snap')
def manual_snap():
    success, frame = cap.read()
    if success:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(snapshot_dir, f"manual_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        return jsonify({"status": "ok", "message": f"📸 Manual snapshot saved: {filename}"})
    else:
        return jsonify({"status": "error", "message": "❌ Camera not ready"})

# -------------------------------
# 🧹 Safe Camera Release
# -------------------------------
def release_camera():
    if cap.isOpened():
        cap.release()
        print("📷 Camera released safely.")

atexit.register(release_camera)


# -------------------------------
# 🚀 Run Flask
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

