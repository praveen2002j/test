// backend/server.js
import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import dotenv from "dotenv";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

const app = express();
app.use(cors());
const upload = multer();

// === Roboflow Model Endpoints ===
const VEHICLE_URL = "https://serverless.roboflow.com/car-detection-ow6wc/6";
const DAMAGE_URL = "https://serverless.roboflow.com/car-damage-detection-5ioys/1";
const API_KEY = "Huxp2j87hU2hw8kiTU9E"; // move to .env later if you want

// === Supabase Storage ===
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || "damageai";

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.warn("⚠️ SUPABASE_URL or SUPABASE_KEY missing from .env");
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// small helpers
const normalize = (x) => x?.toLowerCase().trim();
const safeSlug = (x) => normalize(x || "").replace(/\s+/g, "_");

// generic upload to Supabase
async function uploadToSupabase(buffer, filePath, contentType) {
  const { error } = await supabase.storage
    .from(SUPABASE_BUCKET)
    .upload(filePath, buffer, {
      contentType,
      upsert: false,
    });

  if (error) {
    console.error("❌ Supabase upload error:", error);
    throw error;
  }

  const { data } = supabase.storage
    .from(SUPABASE_BUCKET)
    .getPublicUrl(filePath);

  return data.publicUrl;
}

// =======================
//      UPLOAD ROUTE
// =======================
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No image uploaded",
      });
    }

    // From React Native (FormData)
    // formData.append("carName", "bmw");
    // formData.append("viewIndex", "1");
    // formData.append("viewName", "front");
    // formData.append("sessionId", "1700000000000"); // same for all 4 sides
    const {
      carName = "car",
      viewIndex = "0",
      viewName = "front",
      sessionId,
    } = req.body;

    const originalName = req.file.originalname || "image.jpg";
    const carSlug = safeSlug(carName);        // "bmw"
    const viewSlug = safeSlug(viewName);      // "front"
    const indexStr = String(viewIndex);       // "1".."4"
    const session = sessionId || Date.now().toString(); // fallback if not provided

    console.log("📸 Received:", originalName);
    console.log("🚗 Meta:", { carSlug, viewSlug, indexStr, session });

    const base64Image = req.file.buffer.toString("base64");

    // ================================
    // 1️⃣ VEHICLE DETECTION
    // ================================
    console.log("🚗 Checking for car...");

    const vehicleRes = await axios({
      method: "POST",
      url: VEHICLE_URL,
      params: { api_key: API_KEY, format: "json", confidence: 0.2 },
      data: base64Image,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    const vehiclePreds = vehicleRes.data?.predictions || [];

    const VALID_CAR_CLASSES = [
      "car", "hyundai", "kia", "mazda", "toyota", "jac", "mvm",
      "peugeot", "dena - ikco", "samand - ikco", "tara - ikco",
      "rana - ikco", "quick - saipa", "saina - saipa", "tiba - saipa",
      "pride - saipa", "brilliance", "lifan", "megan - reno", "tondar90 - reno",
      "atlas - saipa", "zantia", "kmc", "sma", "tiggo - chery", "dignity",
      "peykan - ikco", "pars - peugeot", "neissan - zamyad", "206 - peugeot",
      "207 - peugeot", "405 - peugeot", "0-plate"
    ].map(normalize);

    const carBox = vehiclePreds.find(
      (p) => VALID_CAR_CLASSES.includes(normalize(p.class)) && p.confidence >= 0.20
    );

    if (!carBox) {
      console.log("🚫 No car detected.");
      return res.json({
        success: true,
        carDetected: false,
        damageFound: false,
        message: "No car detected",
      });
    }

    console.log("✅ Car detected! Checking damage...");

    // ================================
    // 2️⃣ DAMAGE DETECTION (NEW MODEL)
    // ================================
    const damageJsonRes = await axios({
      method: "POST",
      url: DAMAGE_URL,
      params: { api_key: API_KEY, format: "json", confidence: 0.4 },
      data: base64Image,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    const damagePreds = damageJsonRes.data?.predictions || [];

    const DAMAGE_CLASSES = [
      "front-windscreen-damage",
      "headlight-damage",
      "major-rear-bumper-dent",
      "rear-windscreen-damage",
      "runningboard-dent",
      "sidemirror-damage",
      "signlight-damage",
      "taillight-damage",
      "bonnet-dent",
      "doorouter-dent",
      "doorouter-paint-trace",
      "doorouter-scratch",
      "fender-dent",
      "front-bumper-dent",
      "front-bumper-scratch",
      "medium-bodypanel-dent",
      "paint-chip",
      "paint-trace",
      "pillar-dent",
      "quaterpanel-dent",
      "rear-bumper-dent",
      "rear-bumper-scratch",
      "roof-dent",
    ].map(normalize);

    const filteredDamages = damagePreds.filter(
      (p) => DAMAGE_CLASSES.includes(normalize(p.class)) && p.confidence >= 0.4
    );

    const damageFound = filteredDamages.length > 0;

    console.log(
      damageFound
        ? `⚠️ Damage detected: ${filteredDamages.map((d) => d.class).join(", ")}`
        : "✅ No damage found"
    );

    const IMAGE_SIZE = 1024; // default inference resolution

const damagePoints = filteredDamages.map((d) => ({
  label: normalize(d.class),
  // normalize into 0–1 so we can place as % later
  x: d.x / IMAGE_SIZE,
  y: d.y / IMAGE_SIZE,
}));

    // ================================
    // 3️⃣ ANNOTATED IMAGE (IMAGE MODE)
    // ================================
    const annotatedRes = await axios({
      method: "POST",
      url: DAMAGE_URL,
      params: {
        api_key: API_KEY,
        format: "image",
        labels: "true",
        confidence: 0.4,
      },
      data: base64Image,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      responseType: "arraybuffer",
    });

    const annotatedBuffer = Buffer.from(annotatedRes.data);

    // ================================
    // 4️⃣ UPLOAD FILES (ORIGINAL + DAMAGE)
    // Paths like:
    //   car/bmw/<session>/bmw_1_front.jpg
    //   damage/bmw/<session>/bmw_1_front_damage.jpg
    // ================================
    const baseName = `${carSlug}_${indexStr}_${viewSlug}`; // e.g. bmw_1_front
    const originalPath = `car/${carSlug}/${session}/${baseName}.jpg`;
    const damagePath = `damage/${carSlug}/${session}/${baseName}_damage.jpg`;

    const originalUrl = await uploadToSupabase(
      req.file.buffer,
      originalPath,
      req.file.mimetype
    );

    const annotatedUrl = await uploadToSupabase(
      annotatedBuffer,
      damagePath,
      "image/jpeg"
    );

    // ================================
    // 5️⃣ RESPONSE
    // ================================
    return res.json({
      success: true,
      carDetected: true,
      damageFound,
      carName: carSlug,
      sessionId: session,
      viewIndex: Number(indexStr),
      viewName: viewSlug,
      originalUrl,
      annotatedUrl,
      damageLabels: filteredDamages.map((d) => normalize(d.class)),
      damagePredictions: filteredDamages.length,
      damagePoints,    // for 3D mapping 
    });

  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// =======================
//  HISTORY ROUTE
//  GET /history?carName=bmw  (optional carName)
//  Returns list of damage + original URLs grouped by file name
// =======================
app.get("/history", async (req, res) => {
  try {
    const { carName } = req.query;
    const carSlug = carName ? safeSlug(carName) : null;

    // We list under "damage" folder; inside we have /carSlug/session/...
    const folder = carSlug ? `damage/${carSlug}` : "damage";

    const { data, error } = await supabase.storage
      .from(SUPABASE_BUCKET)
      .list(folder, {
        limit: 1000,
        offset: 0,
        sortBy: { column: "name", order: "asc" },
        recursive: true,
      });

    if (error) {
      console.error("❌ Supabase list error:", error);
      return res.status(500).json({ success: false, error: error.message });
    }

    // each item in `data` has name like "<session>/bmw_1_front_damage.jpg"
    const items = (data || [])
      .filter((obj) => obj.name.toLowerCase().endsWith(".jpg"))
      .map((obj) => {
        const relativePath = `${folder}/${obj.name}`.replace(/\/+/g, "/");

        const { data: annUrlData } = supabase.storage
          .from(SUPABASE_BUCKET)
          .getPublicUrl(relativePath);

        // reconstruct original path
        const originalRelativePath = relativePath
          .replace(/^damage\//, "car/")
          .replace("_damage.jpg", ".jpg");

        const { data: origUrlData } = supabase.storage
          .from(SUPABASE_BUCKET)
          .getPublicUrl(originalRelativePath);

        // parse car/session/view from path
        const parts = relativePath.split("/");
        //  damage / bmw / <session> / bmw_1_front_damage.jpg
        const sessionId = parts.length >= 4 ? parts[2] : null;
        const fileName = parts[parts.length - 1]; // bmw_1_front_damage.jpg
        const baseNoExt = fileName.replace(".jpg", "");
        const baseNoSuffix = baseNoExt.replace("_damage", ""); // bmw_1_front
        const [carSlugFromName, indexStr, viewSlug] = baseNoSuffix.split("_");

        return {
          carName: carSlugFromName,
          sessionId,
          viewIndex: Number(indexStr),
          viewName: viewSlug,
          annotatedUrl: annUrlData.publicUrl,
          originalUrl: origUrlData.publicUrl,
        };
      });

    res.json({ success: true, items });
  } catch (err) {
    console.error("❌ History error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

const PORT = 4000;
app.listen(PORT, "0.0.0.0", () =>
  console.log(`🚀 Backend running on http://0.0.0.0:${PORT}`)
);