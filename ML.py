from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import os
import time
import threading
import atexit
import numpy as np
from ultralytics import YOLO

CAM_INDEX = 2              
SNAPSHOT_DIR = "static/snaps"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
PORT = 5001             
YOLO_CONF = 0.45          
MOTION_DIFF_THRESH = 30000    
AREA_CHANGE_RATIO = 0.35     
FALL_VELOCITY_THRESH = 25.0  
MIN_SNAPSHOT_INTERVAL = 2.0 
app = Flask(__name__)
yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open camera on index {CAM_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
lock = threading.Lock()
annotated_frame = None     
motion_flag = False        
last_snapshot_ts = 0.0
prev_gray = None
prev_boxes = []  
prev_ts = time.time()
def save_snapshot(frame, reason="auto"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{reason}_{timestamp}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"📸 Snapshot saved: {path}")
    return path
def boxes_from_results(results):
    """Extract boxes list from ultralytics result (results[0].boxes)."""
    boxes = []
    if results and len(results) > 0:
        r = results[0]
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].numpy().astype(int)
            x1, y1, x2, y2 = map(int, xyxy)
            label = r.names[cls]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)
            boxes.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2), "centroid": (cx, cy), "area": area})
    return boxes
def processing_loop():
    global annotated_frame, motion_flag, prev_gray, prev_boxes, last_snapshot_ts, prev_ts
    print("🧠 Processing thread started.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame_draw = frame.copy()
        now_ts = time.time()
        results = yolo(frame, conf=YOLO_CONF, imgsz=640, verbose=False)
        boxes = boxes_from_results(results)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            lab = f"{b['label']}:{b['conf']:.2f}"
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_draw, lab, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21,21), 0)
        motion_score = 0
        if prev_gray is None:
            prev_gray = gray_blur.copy()
            motion_score = 0
        else:
            frame_diff = cv2.absdiff(prev_gray, gray_blur)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = int(np.sum(thresh))
            prev_gray = gray_blur.copy()
        major_event = False
        reason = None
        person_present = any(b['label'].lower() == 'person' and b['conf'] > 0.55 for b in boxes)
        if person_present:
            major_event = True
            reason = "person"
        if motion_score > MOTION_DIFF_THRESH:
            if len(boxes) > 0:
                major_event = True
                reason = "big_motion_with_obj"
            else:
                major_event = True
                reason = "big_motion"
        if prev_boxes and boxes:
            for b in boxes:
                same_label_prev = [p for p in prev_boxes if p['label']==b['label']]
                if not same_label_prev:
                    continue
                dists = [np.hypot(b['centroid'][0]-p['centroid'][0], b['centroid'][1]-p['centroid'][1]) for p in same_label_prev]
                idx = int(np.argmin(dists))
                p = same_label_prev[idx]
                dt = now_ts - prev_ts if now_ts - prev_ts > 0 else 0.001
                vy = (b['centroid'][1] - p['centroid'][1]) / dt
                if p['area'] > 0:
                    ratio = abs(b['area'] - p['area']) / (p['area'] + 1e-6)
                else:
                    ratio = 0
                if vy > FALL_VELOCITY_THRESH and ratio > 0.2:
                    major_event = True
                    reason = "falling_object"
                    x1,y1,x2,y2 = b['bbox']
                    cv2.putText(frame_draw, "FALL!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if major_event:
            with lock:
                motion_flag = True
            if time.time() - last_snapshot_ts > MIN_SNAPSHOT_INTERVAL:
                save_path = save = save_path = save_snapshot(frame, reason or "event")
                last_snapshot_ts = time.time()
        else:
            with lock:
                motion_flag = False

        prev_boxes = boxes
        prev_ts = now_ts
        info = f"mscore:{motion_score} boxes:{len(boxes)}"
        cv2.putText(frame_draw, info, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
        with lock:
            annotated_frame = frame_draw.copy()
        time.sleep(0.02)

proc_thread = threading.Thread(target=processing_loop, daemon=True)
proc_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

def frame_generator():
    global annotated_frame
    while True:
        with lock:
            if annotated_frame is None:
                blank = np.zeros((480,640,3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank)
                frame = buffer.tobytes()
            else:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/manual_snap", methods=["POST"])
def manual_snap():
    with lock:
        if annotated_frame is not None:
            path = save_snapshot(annotated_frame, reason="manual")
            return jsonify({"status":"ok", "file": os.path.basename(path)})
    return jsonify({"status":"fail"}), 500

@app.route("/gallery")
def gallery():
    files = sorted(os.listdir(SNAPSHOT_DIR), reverse=True)
    return jsonify(files)
@app.route("/snap/<filename>")
def snap(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)
def release_all():
    try:
        if cap.isOpened():
            cap.release()
            print("📷 Camera released.")
    except Exception as e:
        print("Error releasing camera:", e)

atexit.register(release_all)
if __name__ == "__main__":
    print(f"Starting server on port {PORT}; camera index {CAM_INDEX}")
    app.run(host="0.0.0.0", port=PORT, debug=False)

