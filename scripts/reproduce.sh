#!/usr/bin/env bash
#
# End-to-end reproduction: downloads a real traffic clip and runs every stage
# M1 -> M11, then the evaluation harness.
#
#   ./scripts/reproduce.sh                  # host Python (.venv), embedded stores
#   ./scripts/reproduce.sh --docker         # run the stages inside the api container
#
# Requires for the full run: Neo4j reachable (build-kg) and, for --with-qa,
# an LLM. Stages that need an unavailable service are reported and skipped
# rather than failing the whole script, so a partial environment still
# produces the artefacts it can.
set -euo pipefail

cd "$(dirname "$0")/.."

VIDEO_URL="https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
VIDEO_ID="car_detection"
VIDEO_PATH="data/raw/${VIDEO_ID}.mp4"
START_TIME="2026-08-16T08:30:00"

# WHY 0.16s AND NOT THE ROADMAP'S 1.0s DEFAULT.
# At a 1.0s sampling interval this clip yields 7 detections and ZERO tracks:
# supervision's ByteTrack only auto-activates a track on its very first
# update() call, so every later first-sighting needs a second consecutive hit
# to confirm, and at 1s gaps a car is rarely still in matching range. 0.16s
# gives 48 detections -> 8 tracks. This is set explicitly here, rather than
# changed in configs/pipeline.yaml, because the roadmap's 0.5-1s guidance
# exists to bound M5's VLM cost -- the tradeoff is a real open decision, and
# this script picks the value that makes the demo actually work while leaving
# the project default untouched.
FRAME_INTERVAL="0.16"

USE_DOCKER=0
WITH_QA=0
for arg in "$@"; do
  case "$arg" in
    --docker) USE_DOCKER=1 ;;
    --with-qa) WITH_QA=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$USE_DOCKER" = "1" ]; then
  RUN=(docker compose exec -T api python -m src.cli)
elif [ -x .venv/bin/python ]; then
  RUN=(.venv/bin/python -m src.cli)
else
  RUN=(python -m src.cli)
fi

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# Runs a stage, but treats a failure as non-fatal and keeps going. Used for
# the stages that depend on an external service.
optional() {
  local label="$1"; shift
  if "$@"; then
    return 0
  fi
  printf '\033[33m   skipped: %s failed (is its service running?)\033[0m\n' "$label"
  return 0
}

step "Fetching the sample clip"
mkdir -p data/raw
if [ -f "$VIDEO_PATH" ]; then
  echo "   already present: $VIDEO_PATH"
else
  curl -fsSL -o "$VIDEO_PATH" "$VIDEO_URL"
  echo "   downloaded $(du -h "$VIDEO_PATH" | cut -f1) -> $VIDEO_PATH"
fi

step "M1  ingest"
"${RUN[@]}" ingest "$VIDEO_PATH" --start-time "$START_TIME" --frame-interval-sec "$FRAME_INTERVAL"

# Clips the manifest produced, skipping any with no sampled frames -- a short
# tail clip can exist with zero frames when the video length is not a multiple
# of clip_length_sec, and the per-clip stages have nothing to do with it.
# Read from the host: data/ is bind-mounted, so this is correct in both modes.
CLIPS=$(python3 -c "
import json
manifest = json.load(open('data/outputs/ingest/${VIDEO_ID}_manifest.json'))
print(' '.join(c['clip_id'] for c in manifest['clips'] if c['frames']))
")
echo "   clips with frames: ${CLIPS:-none}"
if [ -z "$CLIPS" ]; then
  echo "No clip has sampled frames -- nothing to process." >&2
  exit 1
fi

for clip in $CLIPS; do
  step "M2  detect   [$clip]";    "${RUN[@]}" detect "$clip" "$VIDEO_ID"
  step "M3  track     [$clip]";   "${RUN[@]}" track "$clip"
  step "M4  associate [$clip]";   "${RUN[@]}" associate "$clip"
  step "M5  attribute [$clip]";   "${RUN[@]}" attribute "$clip"
  step "M6  vote      [$clip]";   "${RUN[@]}" vote "$clip"
done

step "M7  link";    "${RUN[@]}" link "$VIDEO_ID"
step "M8  confirm"; "${RUN[@]}" confirm "$VIDEO_ID"
step "M9  events";  "${RUN[@]}" events "$VIDEO_ID"

step "M10 build-kg (needs Neo4j)"
optional "build-kg" "${RUN[@]}" build-kg "$VIDEO_ID"

step "M11 index"
"${RUN[@]}" index "$VIDEO_ID"

step "M16 evaluate"
if [ "$WITH_QA" = "1" ]; then
  optional "evaluate --run-qa" "${RUN[@]}" evaluate "$VIDEO_ID" --run-qa
else
  "${RUN[@]}" evaluate "$VIDEO_ID"
fi

cat <<EOF

Done. Artefacts:
  data/outputs/                          per-stage JSON
  data/outputs/eval/${VIDEO_ID}_tables.tex   LaTeX results tables

Accuracy tables need annotation:
  ${RUN[*]} evaluate ${VIDEO_ID} --write-template     # then correct it by hand
  ${RUN[*]} evaluate ${VIDEO_ID} --write-qa-template  # then write real answers
EOF
