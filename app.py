from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time

app = Flask(__name__)

# Folder to save snapshots
SNAPSHOT_DIR = 'static/snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Open the camera (index 2 as tested)
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("❌ Camera is not accessible on index 2. Check if it's in use by another application.")

# Force MJPEG to reduce corrupt JPEG warnings
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

motion_detected = False
prev_frame = None

def gen_frames():
    global motion_detected, prev_frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_detected = cv2.countNonZero(thresh) > 5000

            # Save snapshot if motion detected
            if motion_detected:
                filename = os.path.join(SNAPSHOT_DIR, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(filename, frame)

            prev_frame = gray

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/motion_status')
def motion_status():
    return jsonify({'motion': motion_detected})

@app.route('/snapshots')
def list_snapshots():
    files = sorted(os.listdir(SNAPSHOT_DIR))
    files = [f for f in files if f.endswith(".jpg")]
    return jsonify(files)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

