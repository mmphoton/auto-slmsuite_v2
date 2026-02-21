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

## Milestone Sign-off Table

| Milestone | Scope Summary | Required Artifacts (Code/Tests) | Acceptance Gate Result | Implementer Sign | Reviewer Sign | Date |
|---|---|---|---|---|---|---|
| M0 Program bootstrap | Branch + master task + checklist setup | `qa/umbrella_implementation_stream.md`, this checklist | PASS | Codex | Pending | 2026-02-21 |
| M1 Core contracts freeze | Stable contracts + enforced UI->controller boundary | Pending | PENDING |  |  |  |
| M2 Multi-page navigation shell | 8 required pages + dockable/resizable panels | Pending | PENDING |  |  |  |
| M3 Device lifecycle completion | discover/connect/reconnect/release + safe-stop | Pending | PENDING |  |  |  |
| M4 Logging/errors/progress UX | structured logs + diagnostics + progress indicators | Pending | PENDING |  |  |  |
| M5 Pattern parity + schema forms | schema-driven forms + presets + validation | Pending | PENDING |  |  |  |
| M6 Blaze global enforcement | blaze controls + forced composition on all paths | Pending | PENDING |  |  |  |
| M7 Camera controls completion | typed controls + apply/read/revert + temperature monitor | Pending | PENDING |  |  |  |
| M8 Plot workspace completion | interactive plots + pop-outs + export metadata | Pending | PENDING |  |  |  |
| M9 WGS optimization completion | lifecycle controls + convergence + before/after plots | Pending | PENDING |  |  |  |
| M10 Ratio/lattice optimization | N-beam/lattice targets + sim/hw feedback + export | Pending | PENDING |  |  |  |
| M11 Calibration page completion | load/validate/apply + compatibility + metrics | Pending | PENDING |  |  |  |
| M12 Reproducibility and outputs | naming templates + collision policy + full snapshots | Pending | PENDING |  |  |  |
| M13 Hard QA release blocker | expanded matrices + full smoke coverage | Pending | PENDING |  |  |  |

## Release Blocker Checklist (M13 must be PASS)

- [ ] `qa/matrices.py` expanded with full backend-to-GUI parity matrix.
- [ ] simulation/hardware parity matrix completed with explicit statuses.
- [ ] `tests/test_graphical_app_smoke.py` covers all critical flows.
- [ ] all smoke tests pass in CI and local run.
- [ ] every matrix row maps to at least one smoke/integration test.

## Advancement Decision Log

| From Milestone | To Milestone | Allowed? | Evidence | Approved By | Date |
|---|---|---|---|---|---|
| M0 | M1 | YES | Bootstrap docs created and committed on umbrella branch | Pending | 2026-02-21 |
