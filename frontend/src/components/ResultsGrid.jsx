import { ProductCard } from "./ProductCard";

export function ResultsGrid({ results, meta, loading, loadingMsg, onFindSimilar }) {
  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner" />
        <p className="loading-title">{loadingMsg || "Searching Catalog…"}</p>
        <p className="loading-hint">Running Two-Stage Dense Retrieval & CLIP Reranking across 4,681 products…</p>
      </div>
    );
  }

  if (results === null) return null; // nothing searched yet

  if (results.length === 0) {
    return (
      <div className="empty-state">
        <p className="empty-title">No matching products found</p>
        <p className="empty-hint">Try adjusting your keywords, specifying a color, or searching by image.</p>
      </div>
    );
  }

  return (
    <div className="results-section">
      <div className="results-header">
        <div className="results-count">
          Showing <strong>{results.length}</strong> top verified product{results.length !== 1 ? "s" : ""}
        </div>
        {meta?.mode && (
          <div className="results-mode-badge">
            <span className="mode-indicator" />
            Mode: <strong>{meta.mode.toUpperCase()}</strong>
          </div>
        )}
      </div>
      <div className="results-grid">
        {results.map(p => (
          <ProductCard key={p.pid} product={p} onFindSimilar={onFindSimilar} />
        ))}
      </div>
    </div>
  );
}
