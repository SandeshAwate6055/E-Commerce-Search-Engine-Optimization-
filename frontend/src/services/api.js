/**
 * API service layer — all backend communication goes through here.
 * Backend URL is read from VITE_API_BASE_URL environment variable.
 */
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000, // 5 min — Qwen inference can be slow
});

/** GET /health */
export async function fetchHealth() {
  const { data } = await client.get("/health");
  return data;
}

/**
 * POST /search/text
 * @param {string} query
 * @param {number} topK
 * @param {boolean} applyFilters
 */
export async function searchByText(query, topK = 10, applyFilters = true) {
  const { data } = await client.post("/search/text", {
    query,
    top_k: topK,
    text_weight: 1.0,
    apply_filters: applyFilters,
  });
  return data;
}

/**
 * POST /search/image
 * @param {File} imageFile
 * @param {number} topK
 */
export async function searchByImage(imageFile, topK = 10) {
  const form = new FormData();
  form.append("file", imageFile);
  form.append("top_k", String(topK));
  const { data } = await client.post("/search/image", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * POST /search/multimodal
 * @param {string|null} query
 * @param {File|null} imageFile
 * @param {number} topK
 * @param {number} textWeight
 * @param {number} imageWeight
 */
export async function searchMultimodal(
  query,
  imageFile,
  topK = 10,
  textWeight = 0.5,
  imageWeight = 0.5
) {
  const form = new FormData();
  if (query && query.trim()) form.append("text", query.trim());
  if (imageFile) form.append("file", imageFile);
  form.append("top_k", String(topK));
  form.append("text_weight", String(textWeight));
  form.append("image_weight", String(imageWeight));
  const { data } = await client.post("/search/multimodal", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * Build the URL to display a product image served by the backend.
 * image_path from the API looks like: ../data/images/XXXX.jpg
 * Backend serves images at /images/<filename>
 */
export function getProductImageUrl(imagePath) {
  if (!imagePath) return null;
  const filename = imagePath.replace(/\\/g, "/").split("/").pop();
  return `${BASE_URL}/images/${filename}`;
}
