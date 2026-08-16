# data/raw/

Source videos. Gitignored (see `.gitignore`'s `data/raw/*`) — nothing here is
checked in; this file documents provenance so results are reproducible.

## car_detection.mp4

The first real (non-synthetic) clip run through the full pipeline, M1-M11.
Fixed overhead/near-nadir camera over a road, real cars, CC-BY-4.0 licensed.

```
curl -L -o data/raw/car_detection.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4
```

30.2s, 12.5fps, 768x432, from `intel-iot-devkit/sample-videos`.

Run with:
```
python -m src.cli ingest data/raw/car_detection.mp4 --start-time 2026-08-16T08:30:00
python -m src.cli detect clip_000 --visualize
python -m src.cli track clip_000
python -m src.cli associate clip_000
python -m src.cli attribute clip_000
python -m src.cli vote clip_000
python -m src.cli link car_detection
python -m src.cli confirm car_detection
python -m src.cli events car_detection
python -m src.cli build-kg car_detection   # needs Neo4j running
python -m src.cli index car_detection
```

Findings from this run are documented in `configs/pipeline.yaml` and
`src/utils/config.py` (the detector model default) — see git history for the
full writeup. The open item **not yet resolved**: `ingest.frame_sample_interval_sec`
at its roadmap-recommended default (1.0s) produces detections too sparse for
M3's ByteTrack tracker to confirm any track at all (7 detections -> 0 tracks).
A 0.16s interval fixes it (48 detections -> 8 tracks) but multiplies
downstream VLM cost ~6x and isn't yet a considered default change -- pass
`--frame-interval-sec 0.16` (or similar) until this is resolved properly.
