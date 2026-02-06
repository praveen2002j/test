from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2, base64, numpy as np, time, requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODELS
# ===============================
print("🚗 Loading DAMAGE model...")
damage_model = YOLO("best-2.pt", task="detect")
print("✅ Damage model loaded")

# Roboflow CAR detector
ROBOFLOW_URL = "https://serverless.roboflow.com/car-bcsfh/1"
ROBOFLOW_API_KEY = "AU45UQFUHPWna53ZLkkC"

# ===============================
# CONFIG
# ===============================
IOU_THRES = 0.5
MIN_BOX_AREA = 350

# 🔥 CLASS-WISE CONFIDENCE (FIXES SCRATCH)
CLASS_CONF = {
    "dent": 0.35,
    "bumper-dent": 0.45,
    "scratch": 0.22,          # 👈 key fix
    "glass-damage": 0.45,
    "headlight-damage": 0.35,
    "sidemirror-damage": 0.45,
    "taillight-damage": 0.45,
}

# ===============================
# UTIL: split car into 4
# ===============================
def split_into_4(img):
    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return []

    mx, my = w // 2, h // 2
    return [
        (img[0:my, 0:mx], 0, 0),
        (img[0:my, mx:w], mx, 0),
        (img[my:h, 0:mx], 0, my),
        (img[my:h, mx:w], mx, my),
    ]

# ===============================
# API
# ===============================
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        start = time.time()
        contents = await file.read()

        # ---------------------------
        # Decode image safely
        # ---------------------------
        img_np = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Invalid image"}

        H, W = img.shape[:2]
        annotated = img.copy()
        preds = []

        # ===============================
        # 1️⃣ CAR DETECTION (Roboflow)
        # ===============================
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return {"success": False, "error": "Image encode failed"}

        img_b64 = base64.b64encode(buf).decode()

        try:
            rf_res = requests.post(
                ROBOFLOW_URL,
                params={"api_key": ROBOFLOW_API_KEY},
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15
            ).json()
        except Exception as e:
            print("❌ Roboflow error:", e)
            return {"success": False, "error": "Car detector failed"}

        if not rf_res or not rf_res.get("predictions"):
            return {
                "success": True,
                "carDetected": False,
                "damageFound": False,
                "predictions": [],
                "annotated_image": None,
            }

        car = rf_res["predictions"][0]
        cx, cy = int(car["x"]), int(car["y"])
        cw, ch = int(car["width"]), int(car["height"])

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(W, cx + cw // 2)
        y2 = min(H, cy + ch // 2)

        car_crop = img[y1:y2, x1:x2]
        if car_crop.size == 0:
            return {"success": False, "error": "Empty car crop"}

        # ===============================
        # 2️⃣ SPLIT CAR INTO 4
        # ===============================
        tiles = split_into_4(car_crop)

        # ===============================
        # 3️⃣ DAMAGE DETECTION
        # ===============================
        for tile, ox, oy in tiles:
            if tile.size == 0:
                continue

            results = damage_model.predict(
                tile,
                imgsz=640,
                conf=0.20,          # base low, filtered later
                iou=IOU_THRES,
                device="cpu",
                verbose=False
            )

            for r in results:
                if r.boxes is None:
                    continue

                for b in r.boxes:
                    x1t, y1t, x2t, y2t = map(int, b.xyxy[0])
                    conf = float(b.conf)
                    cls = int(b.cls)

                    label = damage_model.names[cls].lower()
                    min_conf = CLASS_CONF.get(label, 0.4)

                    if conf < min_conf:
                        continue

                    gx1 = x1 + ox + x1t
                    gy1 = y1 + oy + y1t
                    gx2 = x1 + ox + x2t
                    gy2 = y1 + oy + y2t

                    if (gx2 - gx1) * (gy2 - gy1) < MIN_BOX_AREA:
                        continue

                    preds.append({
                        "class": label,
                        "confidence": round(conf, 3),
                        "x": int((gx1 + gx2) / 2),
                        "y": int((gy1 + gy2) / 2),
                    })

                    cv2.rectangle(
                        annotated,
                        (gx1, gy1),
                        (gx2, gy2),
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        annotated,
                        f"{label} {conf:.2f}",
                        (gx1, max(gy1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        ok, out = cv2.imencode(".jpg", annotated)
        if not ok:
            return {"success": False, "error": "Annotation encode failed"}

        print("🧠 detections:", len(preds), "⏱", round(time.time() - start, 2), "s")

        return {
            "success": True,
            "carDetected": True,
            "damageFound": len(preds) > 0,
            "predictions": preds,
            "annotated_image": base64.b64encode(out).decode(),
        }

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return {"success": False, "error": str(e)}