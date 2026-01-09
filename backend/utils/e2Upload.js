// backend/utils/e2Upload.js
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import dotenv from "dotenv";
dotenv.config();

const s3 = new S3Client({
  region: "us-east-1",
  endpoint: process.env.E2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.E2_ACCESS_KEY,
    secretAccessKey: process.env.E2_SECRET_KEY,
  },
  forcePathStyle: true, // required for IDrive e2
});

export const uploadToE2 = async (file, keyPath) => {
  if (!keyPath) {
    throw new Error("❌ uploadToE2 requires keyPath (e.g., 'car/file.jpg')");
  }

  const params = {
    Bucket: process.env.E2_BUCKET,
    Key: keyPath,       
    Body: file.buffer,
    ContentType: file.mimetype,
    ACL: "public-read",         
    CacheControl: "no-cache",
  };

  const command = new PutObjectCommand(params);
  await s3.send(command);

  // Return public URL
  return `${process.env.E2_ENDPOINT}/${process.env.E2_BUCKET}/${keyPath}`;
};