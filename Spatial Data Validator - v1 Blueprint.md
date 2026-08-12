---
tags:
  - "-source-"
  - "system"
  - "career-dev"
aliases:
  - "v1 — Spatial Data Quality Validator · Micro-Execution Blueprint"
created: 2026-06-23
status: "🌿 growing"
up: "[[MOC – Projects]]"
---
# v1 — Spatial Data Quality Validator · Micro-Execution Blueprint

> [!abstract] Thesis
> Build one small thing you **write by hand** and can **defend line-by-line**, that proves you can validate messy computer-vision data. Your Apple study-moderation work is the **story** ("I've seen this data fail from the inside"). This project is the **proof** ("and I built the tool that catches it"). Keep those two separate and you stay honest.

**Goal it serves:** land a Summer 2027 data / data-quality / spatial internship. v1 is the door-opener you carry into applications and coffee chats from ~September 2026.

---

## 0 · The rules of engagement

### What v1 is (and why)
A **batch** Python pipeline that ingests images + bounding-box labels, runs a pretrained detector, computes data-quality metrics, deliberately breaks a copy of the data to prove the checks work, stores results in a real database, and routes good vs. bad frames.

> [!warning] What v1 is NOT
> No streaming, no cloud, no Kafka/Flink/Redpanda, no Terraform. Those are senior-grade infra and the wrong fight at 3–4 hrs/week from zero. Batch is what most real data engineering actually is, and a batch pipeline you can explain beats a streaming one you can't. Streaming/cloud is **v2** — and its right home is your CSE 6242 (DVA) group project, where you can offload the deployment.

### The weekly budget
You have **3–4 hours/week. Do one thing at a time.**
- **~1 hr/week → SQL drills**, separate from the project (the project gives SQL *context*; drills give *reps*; you need both).
- **~2–3 hrs/week → the current module**, in order, below.
- Do **NOT** parallelize build + recruit + interview-prep. Sequence them.

### The timing constraint (non-negotiable)
> [!danger] Protect Fall 2026
> CS 7643 (Deep Learning) is one of the most time-intensive courses in OMSA, and it lands on the recruiting window. You have direct evidence of what overload does (the Spring 2026 Optimization D). So:
> - **Core v1 (M1–M6 + ship) must be demoable before DL starts (~late August 2026).**
> - In Fall, recruiting takes the spare hours; DL gets priority; you build **no new features**.
> - **M7 (robustness polish) and the Dagster stretch move to December break.**

### Summer timeline (≈9 weeks, late Jun → late Aug)
| Module           | Focus                      | Est.   |
| :--------------- | :------------------------- | :----- |
| M1               | Repo + data ingestion      | 1 wk   |
| M2               | Hand-rolled quality checks | 1.5 wk |
| M3               | Inference (YOLO26)         | 1 wk   |
| M4               | DriftSimulator ⭐           | 1 wk   |
| M5               | Detect injected drift      | 1 wk   |
| M6               | Store + SQL                | 1.5 wk |
| M8               | Ship (README + Loom)       | 1 wk   |
| **M7 / stretch** | **Deferred → Dec break**   | —      |

If HCI crunches, let M2/M6 stretch — but hold the "demoable before DL" deadline.

---

## 1 · The two principles that govern everything

> [!tip] The reconstruction test (this is what "depth" means)
> After each module: **close the AI, open a blank file, and rebuild the core piece from memory.** If you can't, you learned the *output*, not the *method* — redo it. This is also literally the interview surface: they will ask you to reconstruct and extend your own work.

> [!tip] AI vs. no-AI — the rule of thumb
> **If an interviewer could probe it, you write it. If it's plumbing they'd never ask about, AI can draft it — but you must read and explain every line.** Use AI as a *tutor* and a *plumber*, never as the *author* of the parts you're claiming as proof.

---

## 2 · Build modules

Each module: **Goal · Build steps · Learn · AI split · Question unlocked · Reconstruction gate · Resource.**

### M1 — Repo + data ingestion `(~1 wk)`
**Goal:** a clean repo and a working loader that turns images + labels into records you can inspect.

**Build**
- [x] Repo structure: `/data/{raw,degraded}`, `/src`, `/checks`, `/analysis`, `/outputs/{clean,quarantine,dead_letter,reports}`, `README.md`, `requirements.txt` ✅ 2026-06-26
- [x] `venv` + install deps; `git init`; first commit ✅ 2026-06-26
- [x] Download `coco8` (8 images, auto-downloads via Ultralytics) to verify the loop, then a larger spatial subset (KITTI / BDD100K / COCO subset) ✅ 2026-06-26
- [x] Write a loader → list of records (`frame_id`, `image_path`, `boxes`) using YOLO label format (one `.txt` per image: `class x_center y_center w h`, normalized) ✅ 2026-06-27
- [x] Draw boxes on 3–5 images, save to `/outputs/reports` to eyeball ✅ 2026-06-27

**Learn:** repo hygiene, `pathlib`, how a CV dataset is laid out on disk, what an annotation file contains.
**AI ok:** repo scaffolding, `requirements.txt`, the `cv2.rectangle` drawing loop.
**No AI:** understanding the label format — be able to explain what each number in a YOLO label line means and why it's normalized.
**Question unlocked:** *"How is a CV dataset structured on disk, and what's in an annotation file?"*
**Reconstruction gate:** from a blank file, parse one YOLO label line and draw the box without looking it up.
**Resource:** Ultralytics dataset-format docs; OpenCV basics.

#### What each number in a YOLO label means and why normalized

What each number means: A YOLO label file has one line per object. Each line is five values: `class_id x_center y_center width height.`

class_id — an integer index identifying what the object is (e.g. 0 = person), pointing into a separate ordered list of class names defined in the dataset's .yaml.
x_center, y_center — the center of the bounding box (not a corner), as fractions of image width and height.
width, height — the box's size, also as fractions of image width and height.

Why normalized (the part to sharpen): All four coordinates are stored as fractions between 0 and 1, relative to the image's dimensions, rather than as raw pixels. Your reasoning is right — here's the precise why: it makes the labels resolution-independent. If a training pipeline resizes or crops the image (which it almost always does), pixel coordinates would point to the wrong place, but a fraction like 0.48 is still "48% across" at any size. So the same annotation survives any resize the pipeline applies, and you can mix images of different dimensions in one dataset without per-image bookkeeping.T

It's center+size, not corners. YOLO deliberately stores the center and dimensions, which is why drawing a box requires converting to corners (walk half the width/height out from center — your Part 3 math).
The class is an integer index, the four coordinates are floats. That type distinction is what your loader had to handle.

---

### M2 — Hand-rolled quality checks `(~1.5 wk)`
**Goal:** quantify image and label quality with functions you fully understand.

**Build**
- [ ] `/checks/image_quality.py`: `brightness` (mean of grayscale), `blur` (variance of the Laplacian), `resolution`, optional `contrast`
- [ ] `/checks/label_quality.py`: boxes out of bounds, zero/negative area, invalid class id
- [x] Each returns a **metric** + a **pass/fail** against a threshold; keep thresholds in a config dict/file. Concept: **data contract** (or an _interface_, or a _schema_) `Every check returns a dict: {"metric": metric, "status": True for passed / False for Failed}` ✅ 2026-07-08
- [ ] Run across the dataset → collect metrics into a pandas DataFrame

**Learn:** `numpy`/`cv2` basics; what Laplacian variance measures (high-frequency content → sharpness; blur kills high frequencies → variance drops); brightness as a distribution.
**AI ok:** `cv2` syntax lookup; tutor-style explanation of the Laplacian.
**No AI:** write the blur and brightness functions yourself; be able to explain *why* the metric behaves as it does.
**Question unlocked:** *"How do you measure image quality without a human looking at it?"*
**Reconstruction gate:** re-write the blur detector from scratch and explain the physics in one breath.
**Resource:** PyImageSearch — "Blur detection with OpenCV" (this is exactly the method).

Interview notes:
**`ddepth = cv2.CV_64F` — understand this one, it's interview bait.** The Laplacian produces _negative_ values. Compute it in the input's native 8-bit unsigned type and every negative clips to 0 — you throw away half the edge response and your variance is wrong. Signed float depth preserves the negatives. Be able to say that sentence cold.

---

### M3 — Inference with YOLO26 `(~1 wk)`
**Goal:** attach model signal (detections + confidence) to each frame.

**Build**
- [ ] Load pretrained `yolo26n.pt` (off-the-shelf — you train nothing)
- [ ] Run inference per frame; extract boxes, class labels, confidence scores
- [ ] Append `max_confidence`, `num_detections` to each record
- [ ] Baseline filter: flag frames with `max_confidence < 0.40`

**Learn:** using a model off-the-shelf; what confidence *is* (a model's self-reported certainty) and *isn't* (a guarantee of correctness); YOLO26's NMS-free output.
**AI ok:** the `ultralytics` call boilerplate.
**No AI:** understanding what the output contains; the nuance that a low-confidence frame might be *bad data* OR a *genuinely hard scene* — and how you'd tell.
**Question unlocked:** *"What does a detection model output, and what does confidence actually represent?"*
**Reconstruction gate:** explain what changes in the output if you lower the threshold from 0.40 to 0.10.
**Resource:** Ultralytics docs — quickstart.

---

### M4 — DriftSimulator ⭐ `(~1 wk)` — your signature piece
**Goal:** programmatically degrade a *copy* of the data so you can prove your checks catch failure.

**Build**
- [ ] `/src/drift_simulator.py` — a class that takes a clean dataset and emits a degraded copy
- [ ] Degradations: progressive Gaussian blur (lens fog), brightness decay (failing sensor), label corruption (shift/drop boxes)
- [ ] Parameterize by `severity` (0.0 → 1.0) and the frame index where degradation begins
- [ ] Save to `/data/degraded`

**Learn:** clean, configurable class design; controlled-experiment thinking.
**AI ok:** `cv2` blur/brightness call syntax.
**No AI:** the simulator's design and logic — **own every line.** This is the piece interviewers will dig into.
**Question unlocked:** *"What's the difference between data drift and a corrupted frame, and how would you simulate each?"*
**Reconstruction gate:** rebuild the brightness-decay logic and explain why this module is what *proves* the rest of the pipeline works.

---

### M5 — Detect the drift you injected `(~1 wk)`
**Goal:** show your checks track degradation, and turn that into a PASS/QUARANTINE decision.

**Build**
- [ ] Run M2 checks on clean vs. degraded sets
- [ ] Tabulate: `severity` → metric value → flag rate
- [ ] Tune thresholds so detection tracks severity sensibly (and note where it fails)
- [ ] Assign `validation_status` ∈ {`PASS`, `QUARANTINE`} per frame

**Learn:** baselines, thresholding, sensitivity, what it means to *prove* a detector.
**AI ok:** plotting the severity-vs-detection curve.
**No AI:** the thresholding logic and its interpretation; the experiment design.
**Question unlocked:** *"How do you KNOW your detector works?"* — your ace: a controlled before/after experiment you ran.
**Reconstruction gate:** produce and explain the severity-vs-detection-rate table from your own run, including where it breaks.

> [!note] You have extra firepower here
> Your ML coursework (ISYE 6740, 8803) means you can go beyond thresholds to *statistical* distribution-shift if a question opens that door. Optional for the build; useful in the interview.

---

### M6 — Store it + SQL `(~1.5 wk)`
**Goal:** model the results into a queryable table and write the analyses that sell the project.

**Build**
- [ ] Create a DuckDB database; design `frame_quality_events`:
  - `event_id`, `image_uri`, `sensor_id`, `captured_timestamp`, `brightness`, `blur_score`, `max_confidence`, `drift_severity`, `validation_status`, `ingested_at`
- [ ] Write all records in
- [ ] `/analysis/queries.sql`:
  - **Worst batch:** `GROUP BY` sensor/time window, `HAVING` quarantine count or drift above a bar
  - **Best training data:** filter `PASS` + high confidence + low drift, `ORDER BY`, `LIMIT`
  - **Window function:** rolling average confidence over time per sensor

**Learn:** schema design, partitioning intuition, and SQL for real — joins, aggregation, `HAVING`, CTEs, window functions.
**AI ok:** DuckDB connection boilerplate; syntax check **after** you've attempted the query.
**No AI:** **write every query yourself first.** SQL is the single most-screened skill — do not outsource the thing you're being tested on.
**Question unlocked:** *"Why this storage choice, and write me a query that finds the worst-performing batch."*
**Reconstruction gate:** from a blank query window, write the worst-batch and best-data queries without help.
**Resource:** DuckDB docs (Python guide); DataLemur for SQL reps (window functions + CTEs).

---

### M8 — Ship it `(~1 wk)` — do this even if M7 is deferred
**Goal:** make the work legible to a hiring manager.

**Build**
- [ ] `README.md`: problem → architecture (Mermaid diagram) → how to run → results → what you learned
- [ ] Honest resume bullets (see §5)
- [ ] 2-minute Loom demo (one take, no script)
- [ ] Clean repo; pin `requirements.txt`; include a sample dataset or a download script

**Learn:** communicating work — the part most portfolios skip.
**AI ok:** README first-draft prose; Mermaid syntax.
**No AI:** the "why" narrative and your Loom talking points, in your own words.
**Question unlocked:** *"Give me the 2-minute version of what you built and why it matters."*
**Reconstruction gate:** record the Loom in one take without reading a script.

---

### M7 — Route, report, survive bad input `(~1.5 wk · DEFER to Dec break)`
**Goal:** harden the pipeline so it doesn't fall over on real-world mess.

**Build**
- [ ] Route `PASS` → `/outputs/clean`, fail → `/outputs/quarantine`
- [ ] Malformed label file → `/outputs/dead_letter` + log, **don't crash** (validation + `try/except`)
- [ ] Summary report: matplotlib charts (metric distributions, flag rates) or simple HTML — optionally wired to your **HCI dashboard** (see §6)
- [ ] Make the run idempotent (re-running doesn't duplicate or corrupt)

**Learn:** error handling, the dead-letter pattern, idempotency, reporting.
**AI ok:** chart formatting boilerplate; HTML template.
**No AI:** the error-handling logic and a real understanding of idempotency.
**Question unlocked:** *"Walk me through your pipeline's failure modes — what happens when a label file is broken?"*
**Reconstruction gate:** feed it a deliberately corrupted file and trace what happens, from your own code.

---

### Stretch — v1.5: Dagster `(Dec break · optional)`
**Goal:** wrap M3–M7 as software-defined assets to add an orchestration story.

- [ ] Do Dagster University "Dagster Essentials" first (free, 6–10 hrs)
- [ ] Convert the pipeline steps into Dagster assets with dependencies
- [ ] **Add the resume bullet only if you actually do it**

**Question unlocked:** *"What does an orchestrator give you that a plain Python script doesn't?"*

---

## 3 · The knowledge you stack (outcome ≠ goal)

The point isn't the repo — it's what you can *do and claim* afterward that you couldn't before:

| After | You can now do / say |
| :-- | :-- |
| M1–M2 | Quantify image quality without a human; explain what "data quality" concretely means for CV |
| M3 | Read a detector's output; explain what confidence is and isn't |
| M4–M5 | Design and run a **controlled** data experiment and prove a detector works — rare, senior-flavored |
| M6 | Model data into a schema and write analytical SQL — the most-screened skill |
| M7–M8 | Build a robust, documented, communicable pipeline — the part most portfolios skip |

Each rung is a **transferable capability** that outlives this project.

---

## 4 · The questions you'll answer in depth by the end

This is the real deliverable — interview ammo you can't answer today but will own from your own code:

1. What does "bad data" mean for a CV dataset, and how do you detect it programmatically?
2. How do you measure image quality without a human in the loop?
3. What does a detection model output, and what does confidence mean?
4. What's the difference between drift and corruption, and how would you simulate each?
5. How do you *prove* a data-quality check actually catches what it's supposed to?
6. Why batch here — and when would you genuinely need streaming instead?
7. Why this storage format, and how would you query for the worst batch / best training data?
8. What are your pipeline's failure modes when input is malformed?
9. If this had to run on **10 million** frames instead of 10 thousand, what breaks first and what would you change?

> [!example] Question 9 is your bridge to v2
> You haven't built streaming or cloud — but answering "what breaks at 10M frames" intelligently turns that gap into a *conversation* instead of a hole. It's also the natural setup for pitching the cloud version as your DVA group project.

---

## 5 · AI vs. no-AI — master split

| Do it yourself (no AI authoring) | AI is fine (plumbing / tutor) |
| :-- | :-- |
| Image-quality check logic (M2) — the math *and* the why | Repo scaffolding, `requirements.txt`, Dockerfile (M1) |
| DriftSimulator core (M4) — your signature piece | API glue: calling Ultralytics, connecting DuckDB |
| Every SQL query (M6) — write first, AI checks *after* | Debugging help — understand the fix, don't blind-paste |
| Every design decision + its justification | Concept explanations ("explain Laplacian variance") — then *you* implement |
| The reconstruction gates | Boilerplate plotting / report formatting (M7) |
| Your Loom talking points + README "why," in your words | First-draft README prose — then edit for truth |

> [!warning] The trap
> If AI writes the M4 simulator or your M6 queries, you've hollowed out exactly the two things the interview digs into. Tutor and plumber — never author.

---

## 6 · Synergies to exploit (free leverage)

- **HCI (CS 6750, now):** use the design project to build the **dashboard a study moderator would use to review flagged anomalies** — that's v1's reporting layer (M7). Two deliverables, one effort.
- **Deep Learning (CS 7643, Fall):** v1 uses YOLO26 as a black box; DL teaches you what's *inside* it. Sequence, not gap: "used the detector in my project, then learned its internals in 7643" is a strong, honest arc that deepens your spring interviews for free.
- **ML theory (6740, 8803):** you can answer the drift/distribution questions with more statistical depth than v1 assumes.

---

## 7 · Tools, data, commands

**Install**
```
pip install ultralytics opencv-python numpy pandas duckdb matplotlib
# optional later: deepchecks, dagster
```

**Stack:** Python 3.10+, `ultralytics` (YOLO26), `opencv-python`, `numpy`, `pandas`, `duckdb`, `matplotlib`, `git`.

**Data:** start tiny with `coco8` (auto-downloads, verifies the loop), then a **KITTI** or **BDD100K** subset for the spatial story; **Roboflow Universe** is the easiest source of pre-labeled sets. Get it working small, then scale — that's the correct engineering instinct anyway.

---

## 8 · Honest resume bullets (final output)

- Built a Python data-quality pipeline that validates computer-vision datasets — detecting blur, exposure, and label anomalies — and routes failures to quarantine.
- Engineered a configurable fault-injector simulating sensor degradation (progressive blur, exposure decay, label corruption) to validate detector sensitivity through controlled experiments.
- Modeled validation results in DuckDB and authored analytical SQL (window functions, CTEs) to surface degraded batches and extract high-confidence training candidates.
- *(Stretch, only if done)* Orchestrated the pipeline as Dagster software-defined assets.

> [!warning] No fabrication
> No "sub-millisecond latency," no "edge-to-cloud," nothing you can't reconstruct on a whiteboard.

---

## 9 · Resources (checked, free where it matters)

| Need | Resource |
| :-- | :-- |
| SQL fundamentals | SQLBolt (free, interactive) |
| SQL interview reps | DataLemur free SQL tutorial + free-tier questions (CTEs, window functions) |
| Python / pandas reference | Wes McKinney, *Python for Data Analysis* (free online) |
| M2 blur detection | PyImageSearch — "Blur detection with OpenCV" |
| M3 detector | Ultralytics docs — YOLO26 quickstart (`docs.ultralytics.com`) |
| M6 storage + SQL engine | DuckDB docs — Python guide |
| Stretch orchestration | Dagster University — "Dagster Essentials" (free, 6–10 hrs) |
| Data | Ultralytics `coco8`; KITTI / BDD100K subset; Roboflow Universe |
| Optional | Deepchecks docs (verify it's still maintained before investing — otherwise hand-roll, better for depth) |

---

## Connected ideas
- [[Data-Centric AI & Spatial Engineering Path (2026 Edition)]]
- [[4-Month Cadence - Waterloo Co-op Emulation]] *(macro strategy — separate file, to build next)*
- [[DE Progress]]
