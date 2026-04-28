# Production Sentiment Analysis Web App

## Folder Structure

```text
.
+-- backend
|   +-- .env.example
|   +-- main.py
|   +-- Procfile
|   +-- render.yaml
|   +-- requirements.txt
+-- frontend
    +-- .env.example
    +-- index.html
    +-- package.json
    +-- public
    |   +-- manifest.json
    |   +-- sw.js
    |   +-- icons
    |       +-- icon.svg
    +-- src
    |   +-- App.jsx
    |   +-- main.jsx
    |   +-- styles.css
    +-- vercel.json
    +-- vite.config.js
```

## Backend: FastAPI on Render

Local run:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Render settings:

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english`
  - `ALLOWED_ORIGINS=https://your-vercel-app.vercel.app`

The API exposes:

- `GET /health`
- `POST /predict` with body `{ "text": "..." }`

## Frontend: React on Vercel

Local run:

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Set `VITE_API_URL` in `.env` locally and in Vercel project environment variables:

```text
VITE_API_URL=https://your-render-api.onrender.com
```

Vercel settings:

- Framework preset: Vite
- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`

## PWA and Offline Behavior

- `public/manifest.json` enables installability and Add to Home Screen.
- `public/sw.js` caches the app shell and static assets.
- API responses are cached in `localStorage` by input text.
- If the browser is offline, the UI returns the cached response for matching text when available.
