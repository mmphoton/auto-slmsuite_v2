# GUI Capability Checklist (Sequential Hard Gates)

This checklist is the signed gate record for the umbrella stream defined in:
`qa/umbrella_implementation_stream.md`.

**Umbrella branch:** `graphical-app-umbrella-rebuild`

## Gate Rules (must hold for every milestone)

- Milestones execute strictly in order (1 -> 13).
- No overlap unless explicitly dependency-safe and pre-approved in writing.
- Milestone status is binary: `PASS` or `FAIL` (no partial done).
- Failed gate requires fix-in-place before proceeding.
- Each milestone requires: code + tests + signed result.


## Single-Task Execution Directive

All remaining GUI readiness work is condensed into one coordinated execution task: **GUI readiness closure**.
The task is complete only when matrix coverage, milestone gates, and smoke tests all pass together in one verification run.

## Milestone Sign-off Table

| Milestone | Scope Summary | Required Artifacts (Code/Tests) | Acceptance Gate Result | Implementer Sign | Reviewer Sign | Date |
|---|---|---|---|---|---|---|
| M0 Program bootstrap | Branch + master task + checklist setup | `qa/umbrella_implementation_stream.md`, this checklist | PASS | Codex | Pending | 2026-02-21 |
| M1 Core contracts freeze | Stable contracts + enforced UI->controller boundary | `app/state.py`, `app/interfaces.py`, `app/controller.py`, smoke/contracts matrix | PASS | Codex | Pending | 2026-02-21 |
| M2 Multi-page navigation shell | 8 required pages + dockable/resizable panels | `ui/main_window.py`, smoke checks for page labels/layout persistence | PASS | Codex | Pending | 2026-02-21 |
| M3 Device lifecycle completion | discover/connect/reconnect/release + safe-stop | `devices/manager.py`, device lifecycle smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M4 Logging/errors/progress UX | structured logs + diagnostics + progress indicators | controller state logs/progress + smoke coverage | PASS | Codex | Pending | 2026-02-21 |
| M5 Pattern parity + schema forms | schema-driven forms + presets + validation | `app/patterns.py`, `ui/pattern_form.py`, parity smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M6 Blaze global enforcement | blaze controls + forced composition on all paths | controller blaze composition checks + smoke tests | PASS | Codex | Pending | 2026-02-21 |
| M7 Camera controls completion | typed controls + apply/read/revert + temperature monitor | camera schema + settings/telemetry smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M8 Plot workspace completion | interactive plots + pop-outs + export metadata | plotting interaction/export checks + pop-out metadata checks | PASS | Codex | Pending | 2026-02-21 |
| M9 WGS optimization completion | lifecycle controls + convergence + before/after plots | optimization controls/history export smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M10 Ratio/lattice optimization | N-beam/lattice targets + sim/hw feedback + export | ratio-target optimization smoke checks + metadata export | PASS | Codex | Pending | 2026-02-21 |
| M11 Calibration page completion | load/validate/apply + compatibility + metrics | calibration compatibility/apply smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M12 Reproducibility and outputs | naming templates + collision policy + full snapshots | output naming/snapshot smoke checks | PASS | Codex | Pending | 2026-02-21 |
| M13 Hard QA release blocker | expanded matrices + full smoke coverage | `qa/matrices.py`, `tests/test_graphical_app_smoke.py` full pass | PASS | Codex | Pending | 2026-02-21 |

## Release Blocker Checklist (M13 must be PASS)

- [x] `qa/matrices.py` expanded with full backend-to-GUI parity matrix.
- [x] simulation/hardware parity matrix completed with explicit statuses.
- [x] `tests/test_graphical_app_smoke.py` covers all critical flows.
- [x] all smoke tests pass in CI and local run.
- [x] every matrix row maps to at least one smoke/integration test.

## Advancement Decision Log

| From Milestone | To Milestone | Allowed? | Evidence | Approved By | Date |
|---|---|---|---|---|---|
| M0 | M1 | YES | Bootstrap docs created and committed on umbrella branch | Pending | 2026-02-21 |
| M1 | M2 | YES | `milestone_gate_report` and smoke suite PASS | Pending | 2026-02-21 |
| M2 | M3 | YES | multipage shell + layout persistence checks PASS | Pending | 2026-02-21 |
| M3 | M4 | YES | device lifecycle checks PASS | Pending | 2026-02-21 |
| M4 | M5 | YES | structured logs/progress checks PASS | Pending | 2026-02-21 |
| M5 | M6 | YES | schema parity checks PASS | Pending | 2026-02-21 |
| M6 | M7 | YES | blaze enforcement checks PASS | Pending | 2026-02-21 |
| M7 | M8 | YES | camera controls/telemetry checks PASS | Pending | 2026-02-21 |
| M8 | M9 | YES | plotting checks PASS | Pending | 2026-02-21 |
| M9 | M10 | YES | optimization controls/outputs checks PASS | Pending | 2026-02-21 |
| M10 | M11 | YES | ratio target checks PASS | Pending | 2026-02-21 |
| M11 | M12 | YES | calibration checks PASS | Pending | 2026-02-21 |
| M12 | M13 | YES | reproducibility/snapshot checks PASS | Pending | 2026-02-21 |
