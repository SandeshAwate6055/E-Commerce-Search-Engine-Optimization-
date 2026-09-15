import { useRef } from "react";

const TRENDING_SEARCHES = [
  "Men black formal shirt",
  "Women purple cotton leggings",
  "Silk thread bangles",
  "Turquoise oval ring",
  "Gold pearl spike necklace"
];

export function SearchBar({
  query, onQueryChange,
  imagePreview, onImageSelect, onImageClear,
  onSearch, onReset,
  loading, mode,
}) {
  const fileInputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const err = onImageSelect(file);
      if (err) return;
    }
  }

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) {
      const err = onImageSelect(file);
      if (err) return;
    }
    e.target.value = "";
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !loading) onSearch();
  }

  return (
    <div className="search-bar">
      {/* Main Unified Search Row */}
      <div className="search-input-row">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder='Search 4,681 products by style, color, cut, or upload an image…'
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            type="button"
            className="camera-upload-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload an image to search visually"
          >
            📷 {imagePreview ? "Change Photo" : "Photo Search"}
          </button>
        </div>

        <button
          className="btn btn-primary"
          onClick={onSearch}
          disabled={loading || (!query.trim() && !imagePreview)}
        >
          {loading ? "Searching…" : "Search"}
        </button>

        {(query || imagePreview) && (
          <button className="btn btn-secondary" onClick={onReset} disabled={loading}>
            Clear
          </button>
        )}
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* Upload Chip Row if an Image is Attached */}
      {imagePreview && (
        <div className="attached-image-bar">
          <div className="attached-preview-box">
            <img src={imagePreview} alt="Attached query" className="attached-thumb" />
            <div className="attached-info">
              <span className="attached-title">Attached Visual Reference</span>
              <span className="attached-desc">Will be combined with text for multimodal ranking</span>
            </div>
            <button type="button" className="remove-image-btn" onClick={onImageClear} title="Remove image">
              ✕ Remove
            </button>
          </div>
        </div>
      )}

      {/* Quick Discovery / Trending Chips */}
      <div className="search-suggestions">
        <span className="suggestions-label">Popular queries:</span>
        <div className="suggestions-list">
          {TRENDING_SEARCHES.map(s => (
            <button
              key={s}
              type="button"
              className="suggestion-chip"
              onClick={() => {
                onQueryChange(s);
              }}
              disabled={loading}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
