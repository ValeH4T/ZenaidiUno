# Shadowing

YouTube video → sentence-by-sentence shadowing practice.

- `space/` — the Hugging Face Space (server + the original page). Live: https://zenaidiuno-shadowing-v2.hf.space
- `docs/index.html` — the new page (GitHub Pages; also served by the Space at `/v3`).
- Every push to `main` deploys `space/` (and `docs/index.html` as `v3.html`) to the Space through
  `.github/workflows/deploy-hf.yml`, using the repository secret `HF_TOKEN`.

Personal use only.
