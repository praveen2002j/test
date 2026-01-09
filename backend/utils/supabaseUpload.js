// backend/utils/supabaseUpload.js
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";
dotenv.config();

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const bucketName = process.env.SUPABASE_BUCKET;

if (!supabaseUrl || !supabaseKey || !bucketName) {
  console.warn("⚠️ Supabase env vars missing. Check SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_BUCKET");
}

export const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: { persistSession: false },
});

export async function uploadToSupabase(file, path) {
  if (!file || !file.buffer) {
    throw new Error("File buffer is missing in uploadToSupabase");
  }

  const contentType = file.mimetype || "image/jpeg";

  // upload file
  const { data, error } = await supabase.storage
    .from(bucketName)
    .upload(path, file.buffer, {
      cacheControl: "3600",
      upsert: false,
      contentType,
    });

  if (error) {
    console.error("❌ Supabase upload error:", error);
    // this will make your catch block print a clear error instead of "Unexpected token 'I'"
    throw error;
  }

  // get public URL
  const { data: publicData } = supabase.storage
    .from(bucketName)
    .getPublicUrl(path);

  return publicData.publicUrl;
}