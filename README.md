# MemoryNav

**Memory-Augmented Spatial Intelligence System for Indoor Navigation Assistance**

> A real-time indoor navigation assistant for visually impaired and elderly users — edge AI, personalized spatial memory, offline voice guidance.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![Next.js](https://img.shields.io/badge/Next.js-frontend-black)
![YOLOv8](https://img.shields.io/badge/YOLOv8--nano-Ultralytics-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Inspired by **WalkVLM** (2024) · **VISA** (2025) · **VIALM Survey** (2024) · **NavSpace** (ICRA 2026) — see [Research Motivation](#5-research-motivation).

---

### Contents

[Demo](#2-demo) · [The Problem](#3-the-problem) · [Novel Contribution](#4-novel-contribution) · [Research Motivation](#5-research-motivation) · [Architecture](#6-architecture) · [Results](#7-results) · [Privacy](#8-privacy-architecture) · [Tech Stack](#9-tech-stack) · [Installation](#10-installation) · [Usage](#11-usage) · [Limitations](#12-limitations) · [Future Work](#13-future-work)

---

## 2. Demo

<!--
  TODO: replace this with your real demo GIF or video before publishing.
  Recommended per the design doc: chest-lanyard mount, 60-90 seconds,
  showing bounding boxes + a caption of what was spoken (no audio needed
  for the GIF version — captions read faster on GitHub/Twitter anyway).
-->
<p align="center">
  <img src="docs/demo.gif" alt="MemoryNav live demo — detection, risk scoring, and spoken alert" width="720">
  <br>
  <em>Demo GIF goes here — record with the phone chest-mounted, facing forward (see Form Factor note in Usage).</em>
</p>

[▶ Full demo video](docs/demo.mp4) _(add link once recorded)_

---

## 3. The Problem

Falls are the leading cause of injury death among elderly people worldwide, and a large share happen indoors — in the home, the place people feel safest and where they're least likely to wear corrective glasses consistently.

Existing solutions all have the same gap:

| Solution                      | Critical Gap                                                              |
| ----------------------------- | ------------------------------------------------------------------------- |
| Be My Eyes / Seeing AI / Aira | Requires internet and a human operator or cloud inference                 |
| Generic VLM assistants        | Verbose, causes alert fatigue, repeats the same warning every second      |
| Custom hardware devices       | Expensive, inaccessible, requires technical setup                         |
| **All of the above**          | **None retain persistent memory of the user's specific home environment** |

> **Research backing:** VISA (2025, MDPI) specifically identifies the need for a holistic multi-level approach combining detection, depth, OCR, and spatial awareness for indoor assistive navigation — exactly this project's scope.

**Research question:** _Can personalized spatial memory combined with temporal-aware alert suppression significantly improve indoor navigation assistance for visually impaired users, compared to vision-only detection systems?_ The [ablation study](#7-results) below measures this directly.

---

## 4. Novel Contribution

MemoryNav combines real-time edge perception, personalized spatial memory retrieval, and temporal-aware alert suppression to deliver **context-aware** indoor navigation assistance — not generic object detection. Unlike existing systems, it:

- Remembers the user's specific home layout across sessions (ChromaDB + sentence-transformers)
- Suppresses redundant alerts using WalkVLM-inspired temporal suppression logic
- Runs fully offline on consumer hardware (Apple M2, MPS backend)

This combination — persistent personalized memory _and_ alert-fatigue mitigation, running fully on-device — has not been demonstrated together in prior open-source student-grade work.

---

## 5. Research Motivation

| Paper                                      | What it contributes                                                                                                                                                                                                                                        | Link                                                               |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **WalkVLM** (Yuan et al., Dec 2024)        | Identifies two major usability failures in assistive VLM systems — excessive speech and repeated identical warnings — and proposes temporal suppression. MemoryNav's Alert Manager directly implements this idea.                                          | [arXiv:2412.20903](https://arxiv.org/abs/2412.20903)               |
| **VIALM Survey** (Zhao et al., 2024)       | Surveys LLM-based visually-impaired assistance and finds that LLM outputs are frequently not well grounded in the physical scene. This motivates MemoryNav's rule-based perception core, with the LLM layer kept strictly optional.                        | [arXiv:2402.01735](https://arxiv.org/abs/2402.01735)               |
| **VISA** (2025, MDPI _Journal of Imaging_) | Proposes a holistic, multi-level indoor assistance system combining detection, depth, and text recognition — the structural blueprint MemoryNav's layered pipeline follows.                                                                                | [mdpi.com/2313-433X/11/1/9](https://www.mdpi.com/2313-433X/11/1/9) |
| **NavSpace** (Yang et al., ICRA 2026)      | Benchmarks spatial intelligence and reasoning in embodied navigation agents, establishing personalized/contextual spatial memory as an open frontier problem. Frames MemoryNav's [Future Work](#13-future-work) direction toward semantic spatial mapping. | [arXiv:2510.08173](https://arxiv.org/abs/2510.08173)               |

---

## 6. Architecture

Camera to voice output, every layer named. The vision pipeline (Layers 2–5) is the reliable core and never depends on the optional LLM layer (10) to function.

```mermaid
flowchart TD
    A["1. Camera Frame<br/>~30 FPS, chest-mounted"] --> B["2. Frame Quality Check<br/>blur / brightness gate"]
    B --> C["3. YOLOv8-nano<br/>object detection"]
    C --> D["4. Depth-Anything<br/>monocular distance"]
    D --> E["5. Risk Engine<br/>distance × motion × context"]
    E --> F["6. Short-Term Memory<br/>last 30s session state"]
    E --> G["7. Long-Term Memory<br/>ChromaDB home layout"]
    F --> H["8. Alert Manager<br/>WalkVLM-inspired suppression"]
    G --> H
    H --> I["9. Voice Output<br/>pyttsx3 / ElevenLabs"]
    C -.optional, on-demand.-> J["10. LLM Layer<br/>GPT-4o Vision"]
    J -.-> I
```

**Six core modules**, each with a single clear responsibility:

| #   | Module                   | Function                                                                                                                                              |
| --- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Perception**           | YOLOv8-nano detection + Depth-Anything distance estimation + EasyOCR text reading, gated by a frame-quality check                                     |
| 2   | **Risk Engine**          | `risk = (1 / distance) × motion × context` — turns raw detections into an actionable LOW/MEDIUM/HIGH score                                            |
| 3   | **Memory (RAG)**         | ChromaDB + sentence-transformers retrieve the most relevant piece of home context for what's currently in frame                                       |
| 4   | **Alert Manager**        | Temporal suppression (4s default window) — speaks only if the object changed, the distance bucket changed, or the window elapsed on a HIGH-risk event |
| 5   | **Voice Interface**      | Whisper for voice input ("what's in front of me?"), pyttsx3/ElevenLabs for output, with confidence gating to avoid hallucinated warnings              |
| 6   | **LLM Layer (optional)** | GPT-4o Vision answers complex on-demand questions only — never makes navigation decisions                                                             |

---

## 7. Results

> **Methodology note:** this table is generated by [`evaluation/run_ablation.py`](evaluation/run_ablation.py) and mirrors [`evaluation/results.json`](evaluation/results.json) exactly. Every cell below is `pending measurement` because the ablation hasn't been run on real recorded test videos yet — this project reports **only measured numbers, never estimates.** Regenerate with:
>
> ```bash
> python evaluation/run_ablation.py --videos-dir evaluation/videos --out evaluation/results.json
> ```

**Ablation study** — navigation success rate (user warned before reaching the obstacle):

| Configuration                           | What's missing                               | Navigation success rate |
| --------------------------------------- | -------------------------------------------- | ----------------------- |
| Baseline A: YOLO only                   | No depth, no memory, no suppression          | pending measurement     |
| Baseline B: YOLO + Depth + Risk         | Distance-aware, but no memory or suppression | pending measurement     |
| **Full System**: + Memory + Suppression | Complete MemoryNav                           | pending measurement     |

**Per-component metrics:**

| Component           | Metric                   | What it measures                                       | Value               |
| ------------------- | ------------------------ | ------------------------------------------------------ | ------------------- |
| Object Detection    | mAP@50                   | Mean average precision on structured test videos       | pending measurement |
| Distance Estimation | MAE (m)                  | Mean absolute error vs. tape-measured ground truth     | pending measurement |
| Alert System        | False Alert Rate (%)     | Unnecessary warnings / total warnings                  | pending measurement |
| Alert System        | Miss Rate (%)            | Obstacles reached without warning                      | pending measurement |
| Alert Suppression   | Redundancy Reduction (%) | Alerts suppressed vs. without suppression              | pending measurement |
| Voice Output        | Response Latency (ms)    | Detection to speech start, `time.time()` measured      | pending measurement |
| Memory Retrieval    | Retrieval Accuracy (%)   | Relevant context retrieved / total, manually annotated | pending measurement |

> Ablation studies are standard methodology in ML research (see WalkVLM, VISA) — presenting one in a portfolio project is uncommon at student level. It shifts the conversation from "what did you build" to "what did you prove."

---

## 8. Privacy Architecture

By 2026, the majority of AI inference is expected to run on-device, and local processing is increasingly a baseline expectation — not a differentiator — for any AI system that touches a personal home environment.

- Camera frames are processed **locally** — no video is transmitted or stored
- Whisper speech recognition runs **on-device** — no audio sent to the cloud
- EasyOCR text recognition is **fully offline** — labels and signs never leave the device
- The ChromaDB vector store is **local** — home layout data never touches external servers
- Cloud services (ElevenLabs TTS, GPT-4o Vision) are **opt-in only** and clearly disclosed
- No user data is logged, sold, or used for model training

This follows Privacy-by-Design principles (Cavoukian, 2009), now formalized under GDPR Article 25.

---

## 9. Tech Stack

| Technology            | Source           | Role                | Notes                                                |
| --------------------- | ---------------- | ------------------- | ---------------------------------------------------- |
| YOLOv8-nano           | Ultralytics      | Object detection    | MPS backend on Apple Silicon, ~30 FPS target         |
| Depth-Anything        | Hugging Face     | Distance estimation | Monocular — no LiDAR required                        |
| EasyOCR               | JaidedAI         | Text reading        | Offline, multi-language                              |
| Whisper               | OpenAI (local)   | Voice input         | Runs on-device, no API cost                          |
| ChromaDB              | Chroma           | Vector store        | Long-term spatial memory                             |
| sentence-transformers | Hugging Face     | Embeddings          | Local, no API cost                                   |
| pyttsx3               | Open source      | Offline TTS         | Zero latency, fully offline                          |
| ElevenLabs            | ElevenLabs API   | Premium TTS         | Optional, natural voice                              |
| FastAPI               | Python           | Backend API         | WebSocket for live frame streaming                   |
| SQLite                | Built-in         | User preferences    | No server required                                   |
| OpenCV                | Open source      | Frame capture       | Webcam / phone camera                                |
| GPT-4o Vision         | OpenAI API       | Optional LLM layer  | On-demand only, never in the navigation path         |
| Next.js               | React            | Frontend            | Live dashboard, setup, preferences, evaluation views |
| Tailwind CSS          | CSS framework    | Styling             | Consistent design system                             |
| Docker                | Containerization | Deployment          | `docker compose up` runs the full stack              |

---

## 10. Installation

### Option A — Docker (recommended, one command)

```bash
git clone https://github.com/<your-username>/memorynav.git
cd memorynav
docker compose up --build
```

Frontend → http://localhost:3000 · Backend → http://localhost:8000

See [`docker-compose.yml`](docker-compose.yml) for the full setup, including the `./memory` volume that persists everything taught to the system across restarts. Note: containerized backend runs on CPU — Docker can't see the host GPU/MPS device, so for full MPS acceleration on Apple Silicon, run the backend natively (Option B) instead.

### Option B — Manual setup

**Backend:**

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh # pulls yolov8n.pt into models/
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
EOF
npm run dev
```

Open http://localhost:3000. Your browser will prompt for camera access on the Dashboard page — allow it.

---

## 11. Usage

**1. Teach MemoryNav your home** — go to **Setup**, describe rooms, furniture, and hazards in plain language ("the kitchen has a low step down from the hallway, on the left"). Each entry is embedded and stored in ChromaDB as a memory node.

**2. Watch the live dashboard** — the **Dashboard** page shows the camera feed with live bounding boxes, the current risk level, the retrieved home-context memory, and an alert log showing exactly why each alert fired or was suppressed.

**3. Tune voice & alerts** — on **Preferences**, set speech speed, language, and alert frequency (all detections / medium-and-up / high-risk only).

**4. Ask a question** — say "what's in front of me?" to trigger Whisper voice input and get an on-demand description (optionally routed through the GPT-4o layer for complex queries).

**5. Run the evaluation** — record test videos, annotate them (schema in [`run_ablation.py`](evaluation/run_ablation.py)), then run:

```bash
python evaluation/run_ablation.py
```

Results land in `evaluation/results.json` and feed the **Evaluation** page and the [Results](#7-results) table above.

**Form factor:** MemoryNav is designed for a chest-mounted phone, camera facing forward — this gives a stable, consistent frame and matches the research setups it's inspired by. Hand-held use is not recommended; camera angle variance significantly degrades depth estimation.

---

## 12. Limitations

Honest about what v1.0 cannot yet do:

- **Not a medical device.** MemoryNav is an assistive tool — not a replacement for mobility aids, white canes, or professional guidance.
- **Performance degrades under poor lighting.** Both depth estimation and object detection rely on adequate ambient light.
- **Transparent objects are a known failure mode.** Glass tables, mirrors, and water surfaces are documented weaknesses of monocular depth models.
- **Requires user validation for safety-critical decisions.** The system augments user judgment; it does not replace it.
- **Indoor-only tested.** Outdoor performance is untested and likely degraded by variable lighting and dynamic scenes.
- **Chest mount required.** Camera angle variability significantly degrades depth estimation quality.

---

## 13. Future Work

| Direction                        | Research connection                 | Why it matters                                                                                                                                                |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Semantic spatial mapping**     | NavSpace (ICRA 2026), MG-Nav (2025) | Move from obstacle-level memory ("chair near sofa") to a room-level home graph with per-room hazard history — enables route planning, not just point warnings |
| **Smart glasses form factor**    | Meta AI Glasses direction (2025)    | Hands-free, socially unobtrusive, removes the chest-mount constraint                                                                                          |
| **Multilingual voice interface** | Whisper multilingual support        | Hindi, Tamil, Marathi — India-specific accessibility reach                                                                                                    |
| **Federated home learning**      | Federated AI research (2025–2026)   | Multiple households improve a shared model without sharing private home data                                                                                  |
| **Wearable vibration feedback**  | Haptic assistive devices research   | Silent alerts for deaf-blind users                                                                                                                            |

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Inspired by: **WalkVLM** ([arXiv:2412.20903](https://arxiv.org/abs/2412.20903)) · **VISA** ([MDPI 2025](https://www.mdpi.com/2313-433X/11/1/9)) · **VIALM Survey** ([arXiv:2402.01735](https://arxiv.org/abs/2402.01735)) · **NavSpace** ([arXiv:2510.08173](https://arxiv.org/abs/2510.08173), ICRA 2026)
