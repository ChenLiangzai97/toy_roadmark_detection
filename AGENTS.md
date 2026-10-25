# AGENTS.md

Pointers for any agent (or human) continuing this repository. Read the
design doc for full contracts; this file is the map.

## Authoritative documents

- Design + acceptance criteria: `docs/ARROW_TOY_DESIGN.md`
- Actual implementation decisions, deviations, failures, evidence:
  `docs/IMPLEMENTATION_NOTES.md` (KEEP THIS CURRENT)

## Non-negotiable coordinate rules (see design doc section 4)

- Ego frame: x forward, y LEFT, z up. Angles in radians. Matrices are
  row-major nested lists named `T_<target>_<source>`.
- Pixel centers at half-integers; principal point defaults [W/2, H/2].
- BEV: row 0 = far (x_max), col 0 = left (y_max); model center output is
  normalized [x_n, y_n], never [row, col].
- Three.js pitfalls already solved here (do not reintroduce):
  mirrored orthographic frustums flip winding; `up` parallel to view
  direction degenerates; readback is bottom-up (flip once).

## Supervision semantics (design 6.5/6.6, regression-tested)

- Ignored arrows (occluded/truncated/too_small/no_visual_evidence) never
  receive background supervision: `allow_no_object=false` blocks the
  no-object target for unmatched queries; all-ignored frames produce
  graph-compatible zero losses; truly empty frames train background.
- Matching is Hungarian on valid GT only; >32 valid arrows must fail.

## Commands (full list in README)

- Build frontend before ANY commit: `npm --prefix web run build`
- Test gate before ANY commit: `python -m pytest tests/ -q`
- Regenerate data: `python -m arrow_toy.render --config configs/data_v1.yaml`
  then audit (must pass) then prepare_cache.
- Training runs write everything needed under `runs/<name>/`; never edit
  result.json by hand; rerun instead.

## Artifact policy

- `data/`, `runs/`, `web/dist/`, `node_modules/` are never committed.
- Splits/sample ids are split-prefixed and globally unique; scene groups
  never cross splits (audited).
- Timing/memory/accuracy numbers must come from real runs; the host GPU is
  degraded (~5-10x) — record that when comparing wall clocks.

## Known open items (2026-10)

- Quick-mode accuracy targets unmet at fixed 1500-step budget. WORKING
  HYPOTHESIS (not proven): sparse-head/matching convergence - evidence
  in docs/REPORT.md (overfit fit, oracle probe, consistent no-SA win).
  Next single-variable experiment per design: ablation E dense head.
- Per-class AP/mAP (101-point), macro-F1 and 11.2 ignore-region
  suppression ARE implemented (arrow_toy/evaluation/metrics.py,
  evaluate CLI); keep their regression tests green when touching them.
