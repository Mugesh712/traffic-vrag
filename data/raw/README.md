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
`src/utils/config.py` — see git history for the full writeup.

**Sampling interval (resolved).** The 1.0s default produced 7 detections and
*zero* tracks: ByteTrack needs a second consecutive hit to confirm a new
identity, and at 1s spacing a moving car has already left matching range.
Swept on this clip:

| interval | frames | detections | tracks | VLM crops | detect (s) |
|---|---|---|---|---|---|
| 1.0 | 32 | 7 | **0** | 0 | 7 |
| 0.5 | 63 | 14 | 4 | 4 | 8 |
| 0.25 | 126 | 31 | 7 | 12 | 10 |
| 0.16 | 189 | 48 | 8 | 24 | 13 |

An earlier note here claimed denser sampling "multiplies VLM cost ~6x" as if
it scaled with frame count. That was wrong: M5 caps crops at
`frames_per_track` per track and M8 at `top_k` per object, so VLM cost tracks
the number and length of *tracks*, and 6x the frames cost under 2x the
detection time. The default is now **0.5s** — the working end of the roadmap's
own 0.5-1s guidance. `reproduce.sh` uses 0.16s, which recovers roughly twice
as many objects and is worth the extra crops for evaluation runs.
