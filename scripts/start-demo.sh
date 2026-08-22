#!/usr/bin/env bash
#
# Bring the stack up on a remote host, pointed at that host's CURRENT public IP.
#
#   ./scripts/start-demo.sh
#
# Why this exists. Two settings must match the address the BROWSER uses, or the
# UI fails in ways that look like a server outage rather than a config problem:
#
#   - VITE_API_BASE_URL -- the frontend's default, localhost:8000, resolves to
#     the viewer's own machine, so every request misses the server entirely.
#   - PIPELINE__API__CORS_ORIGINS -- the API's default allowlist is dev-only, so
#     a public origin is refused even once the URL is right. A wildcard is not
#     an option here: with allow_credentials=True it is a real vulnerability.
#
# On a cloud host with a dynamic address (an AWS Academy Learner Lab reassigns
# one on every restart) hand-editing those after each restart is a step that is
# easy to forget, so this asks the instance for its own address instead.
#
# The tuning below is for a CPU-only box. It is set here rather than in
# configs/pipeline.yaml because the committed defaults are tuned for accuracy
# and should stay that way; this trades some of that for runtime on hardware
# without a GPU. Drop the VLM_* / CONFIRM_TOP_K lines to restore them.
set -euo pipefail
cd "$(dirname "$0")/.."

# IMDSv2: ask the instance metadata service for this host's public address.
# Token-first because IMDSv1 is disabled by default on newer instances.
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
IP=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4)

if [ -z "${IP}" ]; then
  echo "Could not read a public IP from instance metadata." >&2
  echo "Not an EC2 host? Set PUBLIC_API_URL / PUBLIC_CORS_ORIGINS by hand." >&2
  exit 1
fi

cat > .env <<ENVEOF
PUBLIC_API_URL=http://${IP}:8000
PUBLIC_CORS_ORIGINS=["http://${IP}:5173","http://localhost:5173","http://127.0.0.1:5173"]

# qwen2.5:7b (the project default) needs ~4.7GB and does not fit alongside the
# other services on an 8GB host; 3b does.
OLLAMA_MODEL=qwen2.5:3b
# 60s is a GPU-era default. CPU inference, especially the first call that also
# loads the model, exceeds it and surfaces as a 500 from /query.
ANSWER_TIMEOUT_SEC=300.0

# Florence-2 (M5 + M8) dominates pipeline runtime on CPU. Greedy decoding and
# fewer crops cost accuracy -- more attributes come back "uncertain" rather
# than wrong, which is the trade this project already prefers.
VLM_NUM_BEAMS=1
VLM_FRAMES_PER_TRACK=4
VLM_MAX_NEW_TOKENS=64
CONFIRM_TOP_K=2
ENVEOF

echo "public IP: ${IP}"
docker compose up -d

cat <<EOF

UI:     http://${IP}:5173
API:    http://${IP}:8000/docs
Neo4j:  http://${IP}:7474  (neo4j / \${NEO4J_PASSWORD:-traffic-vrag})

These ports must be open in the instance's security group.
EOF
