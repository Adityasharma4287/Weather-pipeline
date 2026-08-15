# Localized Weather Intelligence Pipeline — Reference Implementation

A fully working, tested implementation of the architecture described in
`earth2-localized-forecasting-architecture.md`, built around
[NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio)'s interface
shapes and model zoo (FCN3, CorrDiff, GFS/HRRR/ARCO ERA5/MRMS data sources).

## What is real vs. simulated

This sandbox has **no GPU and no internet access to NOAA/ECMWF/NASA data
endpoints or NVIDIA's model weights**. Everything that doesn't require
those is implemented for real. Everything that does is implemented as a
clearly-labeled, physically-plausible simulation with a documented
**`# SWAP_POINT`** comment showing exactly what to replace to go live on a
GPU machine with network access.

| Component | Status |
|---|---|
| Security: secrets manager, HMAC artifact signing, hash-chained immutable audit log | **Real** |
| Ingestion: auth, rate limiting, audit logging, ACL enforcement | **Real** |
| Ingestion: actual GFS/HRRR/MRMS network fetch | Simulated (synthetic, seeded, physically-plausible fields) — `src/ingestion/secured_data_source.py` |
| Global forecast: caching, signing, audit trail, multi-step rollout structure | **Real** |
| Global forecast: FCN3 neural network inference | Simulated (advection+smoothing) — `src/forecasting/global_model.py` |
| Downscaling: two-network structure, regional covariate conditioning, quantization-aware precision handling, ensemble generation, audit logging | **Real structure**, simulated weights — `src/downscaling/corrdiff_downscaler.py` |
| Verification: RMSE, ACC, CRPS, rank histogram, spread/skill, confusion matrix / CSI | **Real math**, not mocked at all — `src/verification/metrics.py` |
| Bias correction: quantile mapping + learned residual correction | **Real**, closed-form implementation — `src/verification/bias_correction.py` |
| API: JWT auth, scopes, rate limiting, caching, audit logging | **Real** (`src/api/main.py`, `src/api/auth.py`) |
| API: token *issuance* | Dev-only stand-in for a real IdP (Keycloak/Cognito) — see SWAP_POINT in `src/api/auth.py` |

Every simulated data path is seeded and deterministic (same inputs -> same
outputs), so the pipeline is reproducible and fully testable end-to-end.

## Project layout

```
weather-pipeline/
├── requirements.txt
├── pytest.ini
├── Dockerfile                      # multi-stage, non-root, healthcheck
├── docker-compose.yml              # API + persistent audit-log volume
├── .dockerignore
├── .env.example                    # copy to .env before docker compose up
├── static/
│   └── index.html                  # zero-build browser UI, served at "/"
├── run_pipeline.py                 # standalone CLI demo (Stages A-D, no API)
├── interactive_cli.py              # interactive REPL — real forecasts, direct or --api mode
├── src/
│   ├── security/
│   │   ├── secrets_manager.py      # credential resolution by reference
│   │   ├── audit_log.py            # hash-chained, append-only audit log
│   │   └── signing.py              # HMAC artifact signing/verification
│   ├── ingestion/
│   │   └── secured_data_source.py  # Stage A: secured ETL/data source adapter
│   ├── forecasting/
│   │   └── global_model.py         # Stage B: FCN3-pattern global forecast
│   ├── downscaling/
│   │   ├── regional_covariates.py  # Stage C: static local adapters (DEM, land use...)
│   │   └── corrdiff_downscaler.py  # Stage C: CorrDiff-pattern generative downscaler
│   ├── verification/
│   │   ├── metrics.py              # Stage D: RMSE/ACC/CRPS/rank histogram/CSI
│   │   └── bias_correction.py      # Stage D: quantile mapping + residual correction
│   ├── pipeline/
│   │   └── orchestrator.py         # wires Stages A -> D together
│   └── api/
│       ├── auth.py                 # Stage E: JWT auth + rate limiting
│       ├── schemas.py              # Stage E: request/response models
│       └── main.py                 # Stage E: FastAPI app
└── tests/                          # 39 tests, all passing
    ├── test_security.py
    ├── test_ingestion.py
    ├── test_forecasting_downscaling.py
    ├── test_verification.py
    └── test_orchestrator_and_api.py
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest tests/ -q
# 39 passed
```

## Run the standalone end-to-end demo (no server)

```bash
python run_pipeline.py --region metro-demo-v1 --variable t2m --lead-hours 48 --grid 16
```

Prints a stage-by-stage summary (global forecast → downscaled ensemble →
signature check → RMSE/CRPS/spread-skill → bias correction) and writes
`pipeline_report.json`.

## Interactive CLI (real forecasts, no scripting)

Runs the actual pipeline live — every response is a genuinely computed
forecast, not a canned reply.

```bash
# Direct mode — runs Stages A-D in-process, no server needed
python interactive_cli.py

weather> forecast metro-demo t2m 24
weather> forecast coastal-zone tp 12
weather> history
weather> quit
```

Or point it at a running server to exercise the full secured Stage E path
(JWT issuance, auth, rate limiting, caching):

```bash
uvicorn src.api.main:app --reload &
python interactive_cli.py --api --base-url http://127.0.0.1:8000
```

Each `forecast` command prints the model version, signature-verification
result, RMSE/CRPS/spread-skill against the pseudo-truth field, and an ASCII
heatmap preview of the downscaled field so you can see region-to-region and
variable-to-variable differences immediately.

## Run the API server

```bash
uvicorn src.api.main:app --reload
```

Then:

```bash
# 1. Get a token (dev-only issuance)
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"sub":"demo-user","tenant":"acme","scopes":["forecast:read"]}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 2. Request a forecast
curl "http://127.0.0.1:8000/v1/forecast/demo-metro?variable=t2m&lead_hours=24" \
  -H "Authorization: Bearer $TOKEN"

# 3. Health check
curl http://127.0.0.1:8000/v1/health
```

Unauthenticated requests get `401`; tokens missing the `forecast:read`
scope get `403`; repeated identical requests are served from cache.

## Browser UI (no CLI needed)

A minimal browser demo is served straight from the API — no separate
frontend build. Start the server and open `http://127.0.0.1:8000/` in a
browser: pick a region/variable/lead time, click **Get forecast**, and it
authenticates itself, calls the real pipeline, and renders the downscaled
field as a heatmap with the RMSE/CRPS/spread-skill numbers.

## Docker — build, run, and share it so others can use it

### 1. Build the image

```bash
cd weather-pipeline
docker build -t weather-pipeline-api:latest .
```

Multi-stage build (deps compiled in a builder stage, slim runtime image),
runs as a non-root user, has a built-in `HEALTHCHECK` hitting `/v1/health`.

### 2. Configure secrets

```bash
cp .env.example .env
# edit .env — set real values before exposing this beyond your own machine
```

### 3. Run it

**Plain Docker:**
```bash
docker run -d --name weather-api \
  --env-file .env \
  -p 8000:8000 \
  -v weather-audit-log:/data \
  weather-pipeline-api:latest
```

**Or docker compose (recommended — also wires the persistent audit-log volume):**
```bash
docker compose up -d --build
```

Check it's healthy:
```bash
docker ps                                   # STATUS should say "healthy" after ~10s
curl http://localhost:8000/v1/health
```

Open `http://localhost:8000/` in a browser, or hit the API directly (same
`curl`/token flow as the non-Docker instructions above) — nothing changes
about how the API is used, only how it's hosted.

### 4. Let other users use it

Three ways to hand this to someone else, depending on how much you want to
manage:

**A. Share the image via Docker Hub (simplest for others)**
```bash
docker tag weather-pipeline-api:latest <your-dockerhub-username>/weather-pipeline-api:latest
docker login
docker push <your-dockerhub-username>/weather-pipeline-api:latest
```
Anyone can now run it with:
```bash
docker run -d -p 8000:8000 --env-file .env <your-dockerhub-username>/weather-pipeline-api:latest
```

**B. Deploy it to a cloud VM/container service** — same image works on any
Docker-capable host: AWS ECS/App Runner, Google Cloud Run, Azure Container
Apps, a plain VPS with `docker compose up -d`, or Fly.io/Render's
container deploy. Point them at your image (from step A, or have them
build from this repo) and open port 8000 (or put a reverse proxy / TLS
terminator like Nginx or Caddy in front of it — the architecture doc's
Stage E API/perimeter boundary).

**C. Local network sharing** — if people are on the same network as the
host machine, `http://<host-machine-ip>:8000/` is reachable directly; no
image sharing needed.

### Important before exposing this beyond your own machine

- Replace every placeholder in `.env` (signing keys, ingestion credentials)
  with real, randomly-generated secrets — the shipped defaults are dev-only
  and are **not** safe to expose publicly.
- Remove/replace the dev-only `POST /v1/auth/token` endpoint with real
  verification against an actual identity provider (Keycloak/Cognito) — see
  the SWAP_POINT in `src/api/auth.py` — before letting untrusted users
  reach the API, since right now anyone who can reach the API can issue
  themselves a token.
- Put TLS in front of it (a reverse proxy is the easiest way) if it will be
  reachable over the public internet.

## Going to production on a real GPU / real data

Every simulated component has a `# SWAP_POINT` comment in its module
docstring showing exactly what to change:

1. `src/ingestion/secured_data_source.py` → replace `_fetch_raw` with real
   `earth2studio.data.GFS()` / `HRRR()` / `MRMS()` calls.
2. `src/forecasting/global_model.py` → replace `_advect_and_smooth` with
   `earth2studio.models.px.FCN3.load_model(...)` + `run_deterministic(...)`.
3. `src/downscaling/corrdiff_downscaler.py` → replace
   `_regression_mean_field` / `_diffusion_residual_sample` with
   `earth2studio.models.dx.CorrDiff.load_model(...)`.
4. `src/api/auth.py` → remove `issue_token`, verify against a real IdP's
   JWKS endpoint instead of a shared HMAC secret.
5. `src/security/secrets_manager.py` → replace the in-memory `_store` with
   a real AWS Secrets Manager / Vault client.
6. `src/security/audit_log.py` → point `_append_line` at a WORM object
   store instead of a local file.

None of these swaps change any other module's interface — every adapter
was built to match the shape of the real integration point from the start.
