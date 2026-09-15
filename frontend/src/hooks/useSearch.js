import { useState, useCallback } from "react";
import {
  searchByText,
  searchByImage,
  searchMultimodal,
} from "../services/api";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB

export function useSearch() {
  const [query, setQuery]               = useState("");
  const [imageFile, setImageFile]       = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults]           = useState(null);   // null = not searched yet
  const [meta, setMeta]                 = useState(null);   // mode, text_parse, vision_parse
  const [loading, setLoading]           = useState(false);
  const [loadingMsg, setLoadingMsg]     = useState("");
  const [error, setError]               = useState(null);

  /** Determine search mode from current inputs */
  const mode = query.trim() && imageFile
    ? "multimodal"
    : query.trim()
    ? "text"
    : imageFile
    ? "image"
    : null;

  /** Validate a File before accepting it */
  function validateImage(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return `Unsupported file type "${file.type}". Use JPEG, PNG, or WebP.`;
    }
    if (file.size > MAX_FILE_BYTES) {
      return `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 10 MB.`;
    }
    return null;
  }

  /** Select an image and create a local preview URL */
  function selectImage(file) {
    if (!file) {
      setImageFile(null);
      setImagePreview(null);
      return null;
    }
    const err = validateImage(file);
    if (err) {
      setError(err);
      return err;
    }
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setError(null);
    return null;
  }

  function clearImage() {
    setImageFile(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
  }

  /** Clear everything back to initial state */
  function reset() {
    setQuery("");
    setImageFile(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
    setResults(null);
    setMeta(null);
    setError(null);
    setLoading(false);
  }

  /** Run a search based on current inputs */
  const runSearch = useCallback(
    async (topK = 10, textWeight = 0.5, imageWeight = 0.5) => {
      const hasText  = query.trim().length > 0;
      const hasImage = imageFile !== null;

      if (!hasText && !hasImage) {
        setError("Enter a search query or upload an image.");
        return;
      }

      setError(null);
      setResults(null);
      setMeta(null);
      setLoading(true);

      try {
        let data;

        if (hasText && hasImage) {
          setLoadingMsg("Understanding query and image…");
          data = await searchMultimodal(query, imageFile, topK, textWeight, imageWeight);
        } else if (hasText) {
          setLoadingMsg("Understanding your query…");
          data = await searchByText(query, topK);
        } else {
          setLoadingMsg("Analyzing your image…");
          data = await searchByImage(imageFile, topK);
        }

        if (!data || !Array.isArray(data.results)) {
          throw new Error("Malformed API response");
        }

        setResults(data.results);
        setMeta({
          mode:         data.mode,
          result_count: data.result_count,
          text_parse:   data.text_parse   ?? null,
          vision_parse: data.vision_parse ?? null,
        });
      } catch (err) {
        if (err.code === "ECONNABORTED" || err.message?.includes("timeout")) {
          setError("Request timed out. The AI models may still be loading — please try again.");
        } else if (err.message === "Malformed API response") {
          setError("The backend returned an unexpected response. Please check the API logs.");
        } else if (err.response) {
          const detail = err.response.data?.detail ?? JSON.stringify(err.response.data);
          setError(`Server error ${err.response.status}: ${detail}`);
        } else {
          setError("Cannot reach the backend. Make sure the API server is running on port 8000.");
        }
        setResults([]);
      } finally {
        setLoading(false);
        setLoadingMsg("");
      }
    },
    [query, imageFile]
  );

  return {
    query, setQuery,
    imageFile, imagePreview,
    selectImage, clearImage,
    results, meta,
    loading, loadingMsg,
    error,
    mode,
    runSearch,
    reset,
  };
}
