# put-screener sidecar — the FastAPI wrapper around run_pipeline.py.
#
# BUILD CONTEXT IS THE REPO ROOT, not chain-service/. The pipeline resolves
# fixtures/ and output/ as siblings of chain-service/ (see
# run_screen.DEFAULT_CONSTITUENTS), so the container has to reproduce that
# layout:
#
#     docker build -f chain-service/Dockerfile -t put-screener-sidecar .
#
# 3.12-slim matches the droplet's 3.12.3. It deliberately does NOT match
# the local .venv, which is 3.9 — that venv exists to run the test suite on
# this machine and says nothing about the deployment target. api.py is
# written to 3.9-compatible syntax so both work.
FROM python:3.12-slim

# Unbuffered so `docker logs` shows a ten-minute screen's progress as it
# happens rather than in one dump at the end.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, as their own layer: the dependency install is the
# slow step and it should not re-run every time a .py file changes.
COPY chain-service/requirements.txt ./chain-service/requirements.txt
RUN pip install --no-cache-dir -r chain-service/requirements.txt

COPY chain-service/ ./chain-service/
COPY fixtures/ ./fixtures/

# Non-root. `output/` is created and chowned here so the volume mount has
# an owner that can write to it — a root-owned bind mount is the classic
# way this fails on first deploy, silently, at the first checkpoint.
RUN useradd --create-home --uid 10001 screener \
    && mkdir -p /app/output \
    && chown -R screener:screener /app
USER screener

WORKDIR /app/chain-service

# Container-internal only. With `expose:` and no `ports:` in compose,
# Docker never publishes this to the host, so binding 0.0.0.0 here does
# not widen anything: it is reachable on the compose network and nowhere
# else, and the droplet's ufw SSH-only posture is untouched.
EXPOSE 8000

# No curl in slim, so the healthcheck uses the interpreter that is already
# here. /health deliberately ignores secrets — see api.py — so a missing
# screener.env surfaces at /preflight instead of restart-looping the
# container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# --workers 1 IS LOAD-BEARING, not a default left unset. api.py's run
# registry and its one-run-at-a-time lock are process-local state; a second
# worker would mean a second registry, a second lock, and two concurrent
# screens interleaving writes to screen_results.json and run_manifest.json.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
