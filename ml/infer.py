from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2, base64, numpy as np, time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔥 Loading ONNX model...")
model = YOLO("best.onnx", task="detect")
print("✅ Model loaded")

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    start = time.time()
    contents = await file.read()

    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return {"predictions": [], "annotated_image": None}

    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    results = model.predict(
        source=img,
        imgsz=640,
        conf=0.1,        # 🔥 high recall
        iou=0.7,
        agnostic_nms=True,
        max_det=300,
        device="cpu",
        verbose=False
    )

    preds = []
    annotated = img.copy()

    for r in results:
        if r.boxes is None:
            continue

        for b in r.boxes:
            cls = model.names[int(b.cls)].lower()
            conf = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            preds.append({
                "class": cls,
                "confidence": conf,
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2)
            })

            # heatmap
            heatmap[y1:y2, x1:x2] += conf

            # draw bbox
            cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(
                annotated,
                f"{cls} {conf:.2f}",
                (x1, max(y1-8,10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,0,255),2
            )

    # normalize heatmap
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    blended = cv2.addWeighted(annotated, 0.7, heatmap_color, 0.3, 0)

    _, buf = cv2.imencode(".jpg", blended)
    annotated_b64 = base64.b64encode(buf).decode()

    print("🧠 detections:", len(preds), "⏱", round(time.time()-start,2),"s")

    return {
        "predictions": preds,
        "annotated_image": annotated_b64
    }