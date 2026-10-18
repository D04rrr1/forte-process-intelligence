# Forte Process Intelligence

**ForteBank Datathon 2026 — Local, explainable video review for cash-desk operations**

Forte Process Intelligence is a working prototype that accepts CCTV-style `.mov` video, analyzes activity around a cashier's workstation, and returns a machine-readable screening verdict with timestamps and evidence frames. The goal is to help a human reviewer find relevant moments faster — **not** to replace a compliance officer or make legally binding decisions.

## What it does

- **Video analysis:** a local computer-vision pipeline processes video frames without calling external vision models or hosted LLMs.
- **People and context:** person detection, presence events, basic role inference, and temporal reasoning about the workstation.
- **Cash-focused screening:** custom cash-object detection and time-based consistency checks generate candidates for unattended-cash incidents.
- **Explainable output:** event timestamps, textual findings, and up to three JPEG evidence frames encoded as Base64.
- **Two interfaces:** a FastAPI job API and a Streamlit review interface.

The project also contains a **rule catalog** for additional scenarios. A catalog entry is **not** the same as a validated, trained detector. Signals requiring more context remain candidates for human review.

## Architecture

```text
MOV video
   |
   v
Frame decoding / visual detection
   |
   v
People, object, and presence signals
   |
   v
Temporal checks + rule-aware triage
   |
   +--> Diagnostic signals / review candidates
   |
   v
Binary screening verdict + timecode + evidence
   |
   +--> FastAPI JSON / Streamlit interface
```

Key files:

- `api.py` — authenticated video upload and asynchronous analysis jobs.
- `app.py` — Streamlit user interface.
- `src/pipeline.py` — main local analysis pipeline.
- `src/presence_events.py`, `src/auto_roles.py` — presence and role signals.
- `src/rule_engine.py` — rule-aware screening logic.
- `detect_cash_events.py` — cash detection helpers.
- `rules/` — rule definitions/catalog.

## Requirements and local setup

Python 3.10+ is recommended. Model inference can run on CPU; processing speed depends on the device and clip length.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local, **untracked** `.forte_secrets` file:

```bash
FORTE_API_KEY=replace-with-your-own-random-key
```

Start the API from the repository root:

```bash
set -a
source .forte_secrets
set +a
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000/health` or `http://127.0.0.1:8000/docs`.

Start the review UI separately:

```bash
python -m streamlit run app.py
```

The project uses locally stored model weights. Private competition videos, uploads, runtime outputs, and API secrets are **not included** in this public repository. Some pretrained weights may need to be cached locally by their supporting libraries for a fresh installation.

## API contract

Authentication: `X-API-Key` header. Input: `.mov` file as multipart field `file`.

### 1. Submit a video

```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/analyze' \
  -H "X-API-Key: $FORTE_API_KEY" \
  -F 'file=@sample.mov;type=video/quicktime'
```

Example acknowledgement:

```json
{"job_id":"<job_id>","status":"processing"}
```

### 2. Check the job

```bash
curl 'http://127.0.0.1:8000/api/v1/jobs/<job_id>' \
  -H "X-API-Key: $FORTE_API_KEY"
```

Job statuses used by this implementation: `processing`, `done`, `failed`. A completed job (`done`) includes top-level `verdict` (`violation` / `no_violation`) and a `violations` list. Each detected violation contains `type`, `timecode_sec`, `description`, and `evidence_frames` with `timestamp_sec` and `image_base64`. A nested `result` is also returned for compatibility with the UI.

**Do not share your API key or upload sensitive recordings to unauthorized public services.** For the hackathon demonstration, any temporary public tunnel is a network transport to the locally running API; inference remains on the local machine.

## Measured development-set results

A local check on **42 annotated videos** of the stable prototype reported:

| Metric | Value |
|---|---:|
| True positives | 1 |
| False negatives | 21 |
| True negatives | 20 |
| False positives | 0 |
| Recall for violation | 4.55% |
| Overall accuracy | 50.0% |

These numbers are **development-set observations, not independent test-set performance**. The MVP favors conservative confirmation and therefore misses many violation types. A successful end-to-end API example detected unattended cash in video X-26 at approximately 11.87 s and produced three evidence frames. These observations do not establish general reliability.

## Limitations and next steps

- Only a limited subset of rule scenarios is supported by confirmed positive verdicts; many catalog entries are not implemented as reliable detectors.
- False negatives are substantial on the annotated development set.
- Video context, transaction state, and legal applicability cannot be inferred reliably from visual signals alone.
- Next steps: additional training data, per-rule evaluation, calibrated confidence, independent holdout testing, and human review workflows.

## Privacy and scope

The analysis pipeline runs locally. Raw competition videos and API credentials must remain outside the public repository. Evidence must be used only with permission; the prototype is not intended for production banking surveillance without security, privacy, and compliance review.

---

**Built for ForteBank Datathon 2026.**
