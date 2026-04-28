import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const CACHE_PREFIX = "sentiment-cache:";

function cacheKey(text) {
  return `${CACHE_PREFIX}${text.trim().toLowerCase()}`;
}

function readCachedResult(text) {
  const cached = localStorage.getItem(cacheKey(text));
  return cached ? JSON.parse(cached) : null;
}

function writeCachedResult(text, result) {
  localStorage.setItem(
    cacheKey(text),
    JSON.stringify({ ...result, cachedAt: new Date().toISOString() })
  );
}

function formatScore(score) {
  return `${Math.round(score * 1000) / 10}%`;
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isCached, setIsCached] = useState(false);

  const canSubmit = useMemo(() => text.trim().length > 0 && !isLoading, [text, isLoading]);

  async function handleSubmit(event) {
    event.preventDefault();
    const value = text.trim();
    if (!value) {
      setError("Enter text to analyze.");
      return;
    }

    setError("");
    setResult(null);
    setIsCached(false);

    const cached = readCachedResult(value);
    if (!navigator.onLine && cached) {
      setResult(cached);
      setIsCached(true);
      return;
    }

    if (!navigator.onLine) {
      setError("You are offline and no cached result exists for this text.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: value })
      });

      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.detail || "Unable to analyze sentiment.");
      }

      const data = await response.json();
      writeCachedResult(value, data);
      setResult(data);
    } catch (requestError) {
      if (cached) {
        setResult(cached);
        setIsCached(true);
        setError("Live analysis failed, so a cached result is shown.");
      } else {
        setError(requestError.message);
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-labelledby="app-title">
        <div className="intro">
          <p className="eyebrow">Transformer sentiment analysis</p>
          <h1 id="app-title">Sentiment Analysis</h1>
          <p className="summary">
            Analyze text with a production FastAPI backend and keep recent predictions available
            when the app is offline.
          </p>
        </div>

        <form className="analyzer" onSubmit={handleSubmit}>
          <label htmlFor="sentiment-text">Text</label>
          <textarea
            id="sentiment-text"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste a review, comment, or support message..."
            rows="8"
          />
          <button type="submit" disabled={!canSubmit}>
            {isLoading ? "Analyzing..." : "Analyze"}
          </button>
        </form>

        {error && <p className="notice error">{error}</p>}

        {result && (
          <article className="result" aria-live="polite">
            <div>
              <span className="result-label">Label</span>
              <strong>{result.label}</strong>
            </div>
            <div>
              <span className="result-label">Score</span>
              <strong>{formatScore(result.score)}</strong>
            </div>
            {isCached && <p className="notice">Showing an offline cached result.</p>}
          </article>
        )}
      </section>
    </main>
  );
}
