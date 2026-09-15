import { useEffect, useState } from "react";
import { ResultsGrid } from "./components/ResultsGrid";
import { SearchBar } from "./components/SearchBar";
import { fetchHealth } from "./services/api";
import { useSearch } from "./hooks/useSearch";

function InferredFilters({ textParse }) {
  if (!textParse) return null;

  const filters = [];
  if (textParse.gender) filters.push({ label: "Audience", value: textParse.gender });
  if (textParse.color) filters.push({ label: "Color", value: textParse.color });
  if (textParse.category_hint) filters.push({ label: "Category", value: textParse.category_hint });
  if (textParse.brand) filters.push({ label: "Brand", value: textParse.brand });
  if (textParse.semantic_query && textParse.semantic_query !== textParse.original_query) {
    filters.push({ label: "Intent", value: textParse.semantic_query });
  }

  if (filters.length === 0) return null;

  return (
    <div className="inferred-filters-bar">
      <span className="filters-label">✨ AI Extracted Attributes:</span>
      <div className="filters-list">
        {filters.map((f, i) => (
          <span key={i} className="filter-pill">
            <span className="filter-tag-label">{f.label}:</span> <strong>{f.value}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function DeveloperPanel({ meta, results }) {
  const [open, setOpen] = useState(false);

  if (!meta && !results?.length) return null;

  return (
    <footer className="dev-footer-section">
      <button
        className="dev-toggle-btn"
        type="button"
        onClick={() => setOpen((value) => !value)}
      >
        <span>🛠️ {open ? "Hide" : "Inspect"} Backend Model Signals & Vector Metrics</span>
      </button>

      {open && (
        <div className="dev-details-box">
          {meta?.text_parse && (
            <div className="dev-block">
              <h3>Stage 1: Qwen2.5-0.5B Query Parsing</h3>
              <pre>{JSON.stringify(meta.text_parse, null, 2)}</pre>
            </div>
          )}
          {meta?.vision_parse && (
            <div className="dev-block">
              <h3>Vision Alignment Analysis</h3>
              <pre>{JSON.stringify(meta.vision_parse, null, 2)}</pre>
            </div>
          )}
          {results?.length > 0 && (
            <div className="dev-block">
              <h3>Top-5 Two-Stage Similarity Scores (BGE Dense + CLIP Cross-Modal)</h3>
              <pre>
                {JSON.stringify(
                  results.slice(0, 5).map((item) => ({
                    rank: item.rank,
                    pid: item.pid,
                    bge_dense_score: item.text_score,
                    clip_rerank_score: item.clip_score,
                    final_hybrid_score: item.final_score,
                    retrieved_by: item.retrieved_by,
                  })),
                  null,
                  2
                )}
              </pre>
            </div>
          )}
        </div>
      )}
    </footer>
  );
}

function HealthStatus() {
  const [health, setHealth] = useState({ state: "checking", message: "Connecting to Engine..." });

  useEffect(() => {
    let active = true;

    fetchHealth()
      .then((data) => {
        if (!active) return;
        setHealth({
          state: data?.models_loaded ? "ready" : "loading",
          message: data?.models_loaded
            ? `🟢 4,681 Products Indexed (RTX 2050 CUDA)`
            : "API reachable, models loading",
        });
      })
      .catch(() => {
        if (!active) return;
        setHealth({
          state: "offline",
          message: "⚠️ Backend Offline",
        });
      });

    return () => {
      active = false;
    };
  }, []);

  return <span className={`health-pill health-${health.state}`}>{health.message}</span>;
}

export default function App() {
  const {
    query,
    setQuery,
    imagePreview,
    selectImage,
    clearImage,
    results,
    meta,
    loading,
    loadingMsg,
    error,
    mode,
    runSearch,
    reset,
  } = useSearch();

  function handleFindSimilar(product) {
    const searchTerms = [
      product.main_category,
      product.norm_text ? product.norm_text.split("|")[0].trim() : "",
    ]
      .filter(Boolean)
      .join(" ");
    setQuery(searchTerms || product.product_name);
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <div className="brand-badge">Enterprise E-Commerce Multimodal AI</div>
          <h1>Discover Fashion & Lifestyle Products</h1>
          <p className="hero-subtitle">
            Natural language semantic search and visual similarity across 4,681 catalog items
          </p>
        </div>
        <HealthStatus />
      </header>

      <section className="search-panel" aria-label="Search products">
        <SearchBar
          query={query}
          onQueryChange={setQuery}
          imagePreview={imagePreview}
          onImageSelect={selectImage}
          onImageClear={clearImage}
          onSearch={runSearch}
          onReset={reset}
          loading={loading}
          mode={mode}
        />

        <InferredFilters textParse={meta?.text_parse} />
      </section>

      {error && (
        <section className="error-state" role="alert">
          <strong>Search could not be completed.</strong>
          <p>{error}</p>
        </section>
      )}

      <ResultsGrid
        results={results}
        meta={meta}
        loading={loading}
        loadingMsg={loadingMsg}
        onFindSimilar={handleFindSimilar}
      />

      <DeveloperPanel meta={meta} results={results} />
    </main>
  );
}
