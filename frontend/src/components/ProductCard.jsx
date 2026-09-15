import { useState } from "react";
import { getProductImageUrl } from "../services/api";

const FALLBACK = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Crect width='200' height='200' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23aaa' font-size='14'%3ENo image%3C/text%3E%3C/svg%3E";

export function ProductCard({ product, onFindSimilar }) {
  const [imgSrc, setImgSrc] = useState(getProductImageUrl(product.image_path));

  const discount =
    product.retail_price && product.discounted_price
      ? Math.round((1 - product.discounted_price / product.retail_price) * 100)
      : null;

  // Convert raw similarity score (e.g. 0.45 - 0.55) into a consumer-friendly match percentage
  const rawScore = Number(product.final_score ?? product.text_score ?? 0.45);
  const matchPercent = Math.min(99, Math.max(72, Math.round((rawScore / 0.55) * 100)));

  return (
    <div className="product-card">
      <div className="product-card-top">
        <div className="product-rank">#{product.rank}</div>
        <div className="product-match-badge">
          <span className="match-pill">
            <span className="match-dot"></span>
            {matchPercent}% Match
          </span>
          {product.clip_score != null && (
            <span className="visual-match-pill" title="Multimodal visual alignment verified by CLIP">
              Visual Reranked
            </span>
          )}
        </div>
      </div>

      <div className="product-img-wrapper">
        <img
          src={imgSrc || FALLBACK}
          alt={product.product_name || "Product image"}
          className="product-img"
          onError={() => setImgSrc(FALLBACK)}
          loading="lazy"
        />
        {onFindSimilar && (
          <button 
            type="button" 
            className="find-similar-btn"
            onClick={() => onFindSimilar(product)}
            title="Search for visually similar items"
          >
            🔍 Find Similar
          </button>
        )}
      </div>

      <div className="product-info">
        <div className="product-name" title={product.product_name || "Untitled product"}>
          {product.product_name || "Untitled product"}
        </div>

        <div className="product-meta">
          {product.main_category && (
            <span className="product-category">{product.main_category}</span>
          )}
          {product.brand && product.brand !== "Unknown" && (
            <span className="product-brand">{product.brand}</span>
          )}
        </div>

        <div className="product-price">
          {product.discounted_price != null && (
            <span className="price-current">
              ₹{product.discounted_price.toLocaleString("en-IN")}
            </span>
          )}
          {product.retail_price != null && product.retail_price !== product.discounted_price && (
            <span className="price-original">
              ₹{product.retail_price.toLocaleString("en-IN")}
            </span>
          )}
          {discount !== null && discount > 0 && (
            <span className="price-discount">{discount}% off</span>
          )}
        </div>

        {product.raw_caption && (
          <div className="product-ai-caption" title={product.raw_caption}>
            <span className="ai-caption-icon">✨</span> {product.raw_caption}
          </div>
        )}
      </div>
    </div>
  );
}
