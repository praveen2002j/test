import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import dotenv from "dotenv";
import { createClient } from "@supabase/supabase-js";
import FormData from "form-data";

dotenv.config();

const app = express();
app.use(cors());
const upload = multer();

// 🔥 DAMAGE MODEL ONLY
//const DAMAGE_URL = "https://test2-gjo9.onrender.com/predict-file";
const DAMAGE_URL = "https://praveen2501-new-car-damage-detection-api.hf.space/predict-file";
// const DAMAGE_URL = "http://127.0.0.1:8000/predict-file";
console.log("🚨 DAMAGE_URL USED BY BACKEND:", DAMAGE_URL);
// Supabase
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const normalize = (x) => x?.toLowerCase().trim();

// =======================
//      UPLOAD ROUTE
// =======================
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    console.log("\n📥 NEW IMAGE (DAMAGE ONLY)");

    if (!req.file) {
      return res.status(400).json({ success: false, error: "No image uploaded" });
    }

    const { viewIndex = 1, viewName = "front", sessionId } = req.body;
    const session = sessionId || Date.now().toString();

    // ================================
    // 1️⃣ DAMAGE DETECTION (ALWAYS)
    // ================================
    const form = new FormData();
    form.append("file", req.file.buffer, "image.jpg");

    const damageRes = await axios.post(DAMAGE_URL, form, {
      headers: form.getHeaders(),
      timeout: 120000,  //2mins
    });

    const preds = damageRes.data.predictions || [];

    console.log(
      "🧠 DAMAGE MODEL OUTPUT:",
      preds.map((p) => `${p.class}:${p.confidence.toFixed(3)}`)
    );

    // ================================
    // 2️⃣ UPLOAD IMAGES
    // ================================
    const base = `${viewIndex}_${viewName}`;

    await supabase.storage
      .from("damageai")
      .upload(`car/${session}/${base}.jpg`, req.file.buffer, {
        upsert: true,
        contentType: req.file.mimetype,
      });

    const annotatedBuffer = Buffer.from(
      damageRes.data.annotated_image,
      "base64"
    );

    await supabase.storage
      .from("damageai")
      .upload(`damage/${session}/${base}_damage.jpg`, annotatedBuffer, {
        upsert: true,
        contentType: "image/jpeg",
      });

    const originalUrl = supabase.storage
      .from("damageai")
      .getPublicUrl(`car/${session}/${base}.jpg`).data.publicUrl;

    const annotatedUrl = supabase.storage
      .from("damageai")
      .getPublicUrl(`damage/${session}/${base}_damage.jpg`).data.publicUrl;

    // ================================
    // 3️⃣ RESPONSE
    // ================================
    res.json({
      success: true,
      carDetected: true, // forced true (or remove later)
      damageFound: preds.length > 0,
      damageLabels: preds.map((p) => normalize(p.class)),
      damagePredictions: preds.length,
      damagePoints: preds.map((p) => ({
        label: normalize(p.class),
        x: p.x / 640,
        y: p.y / 640,
        confidence: p.confidence,
      })),
      originalUrl,
      annotatedUrl,
    });
  } catch (e) {
    console.error("❌ ERROR:", e.message);
    res.status(500).json({ success: false, error: e.message });
  }
});

// =======================
//      START SERVER
// =======================
app.listen(4000, () =>
  console.log("🚀 Backend running on http://localhost:4000")
);
