#realtime_face_recog_2
import cv2
import numpy as np
import face_recognition
import pickle
import time

# ---------------- Paths ----------------
DNN_PROTO = r"C:\Users\admin\OneDrive\Desktop\Image processing\deploy.prototxt"
DNN_MODEL = r"C:\Users\admin\OneDrive\Desktop\Image processing\res10_300x300_ssd_iter_140000.caffemodel"
KNOWN_FILE = r"C:\Users\admin\OneDrive\Desktop\Image processing\known_faces.pkl"

# ---------------- Load known faces ----------------
with open(KNOWN_FILE, "rb") as f:
    data = pickle.load(f)
KNOWN_ENCODINGS = data["encodings"]
KNOWN_NAMES = data["names"]

# ---------------- DNN detector ----------------
net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

# ---------------- Parameters ----------------
DETECT_EVERY_N = 12
TOLERANCE = 0.55
MIN_FACE_PIX = 60
TRACKER_TIMEOUT = 2.5  # seconds

# ---------------- Tracker factory ----------------
def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    raise RuntimeError("MOSSE tracker not supported")

# ---------------- Utils ----------------
def detect_faces_dnn(frame_small):
    h, w = frame_small.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_small, 1.0, (300, 300),
                                 (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.55:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype(int))
    return boxes

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / float(areaA + areaB - inter + 1e-6)

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)
trackers = []
frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    H, W = frame.shape[:2]

    # Downscale for detection
    small = cv2.resize(frame, (480, int(H * 480 / W)))

    # -------- Update trackers --------
    now = time.time()
    alive = []
    for tr in trackers:
        ok, box = tr["tracker"].update(frame)
        if ok and now - tr["last_seen"] < TRACKER_TIMEOUT:
            x, y, w, h = map(int, box)
            tr["bbox"] = (x, y, x+w, y+h)
            tr["last_seen"] = now
            alive.append(tr)
    trackers = alive

    # -------- Detection step --------
    if frame_count % DETECT_EVERY_N == 0 or not trackers:
        boxes_small = detect_faces_dnn(small)
        sx, sy = W / small.shape[1], H / small.shape[0]

        for (x1, y1, x2, y2) in boxes_small:
            fx1, fy1 = int(x1*sx), int(y1*sy)
            fx2, fy2 = int(x2*sx), int(y2*sy)

            if fx2-fx1 < MIN_FACE_PIX or fy2-fy1 < MIN_FACE_PIX:
                continue

            if any(iou(tr["bbox"], (fx1,fy1,fx2,fy2)) > 0.4 for tr in trackers):
                continue

            face_rgb = cv2.cvtColor(frame[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(face_rgb)
            if not encs:
                continue

            encoding = encs[0]
            matches = face_recognition.compare_faces(KNOWN_ENCODINGS, encoding, TOLERANCE)
            name = "Unknown"
            if True in matches:
                name = KNOWN_NAMES[matches.index(True)]

            tr = create_tracker()
            tr.init(frame, (fx1, fy1, fx2-fx1, fy2-fy1))
            trackers.append({
                "tracker": tr,
                "bbox": (fx1,fy1,fx2,fy2),
                "name": name,
                "encoding": encoding,
                "last_seen": now
            })

    # -------- Draw --------
    for tr in trackers:
        x1,y1,x2,y2 = tr["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, tr["name"], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    fps = 1/(time.time()-prev_time+1e-6)
    prev_time = time.time()
    cv2.putText(frame, f"FPS:{fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Optimized Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
