# ──────────────────────────────────────────────────────────────────────────────
# This file is NOT a web-server and must NOT be used as a Railway deploy target.
#
# Railway service Dockerfiles:
#   Public web service (node-api)      →  node-api/Dockerfile
#   Internal audio service (python)    →  python-service/Dockerfile
#   CLI / local dev utility            →  Dockerfile.cli
# ──────────────────────────────────────────────────────────────────────────────
FROM busybox
RUN echo "ERROR: Do not deploy this Dockerfile. Use node-api/Dockerfile or python-service/Dockerfile." && exit 1
