## One Architectural Decision to Resolve Before Phase 7B

The Gradio app will call the Prediction API. It needs to know the API URL. Two options to decide next session:

**Option H1 — Gradio as a fifth Docker Compose service** (planned from the start): `PREDICTION_API_URL=http://prediction-api:8000` via environment variable. The commented-out `gradio-ui` service in `docker-compose.yaml` is already stubbed and ready to uncomment.

**Option H2 — Gradio runs locally** (simpler for development): `PREDICTION_API_URL=http://localhost:8000`. No Docker image needed until Phase 8.

Both are valid. The decision affects whether Phase 7B delivers a `docker/gradio_ui/Dockerfile` alongside the `src/ui/` module.
