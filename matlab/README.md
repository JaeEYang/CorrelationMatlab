# MATLAB reference implementation (frozen)

This is the original MATLAB Offline Correlation tool by Jae Yang. It is **frozen
reference material** and is being retired.

- No Python code in this repository depends on anything here.
- Do not add features or fix bugs here.
- Its only remaining purpose is to define **golden outputs** that the Python
  implementation must reproduce before MATLAB is retired.

## Entry point

`OfflineCorrelationGUI.m` — the tool as shipped. Its six steps define the offline
workflow the Python application must reach parity with:

| Step | Function | Planned Python replacement |
|---|---|---|
| Load FLM image + points | `loadImage.m`, `loadCSV.m` | `io/images.py`, `io/points_csv.py` |
| Load TEM image + points | same | same |
| Compute transform | `computeTransform.m` | `core/transform.py` |
| Warp FLM into TEM frame | `warpImage.m` | `core/warp.py` |
| Show overlay | `interactiveOverlay.m` | overlay viewer window |
| Save results | `saveResults` (in the GUI) | export flattened overlay |

`transformPoints.m` maps auxiliary points through the fitted transform.

## Conventions used here

These matter, because the Python side deliberately matches them:

- Points are **3×N homogeneous column vectors**, `[x; y; 1]`.
- The transform is applied as `q = M * p` (column-vector convention).
- `computeTransform.m` solves `M = Q * P' * inv(P * P')`.

The Python core uses the same column-vector convention, so the two can be compared
matrix-to-matrix in tests.

## Dead even here

`loadFLMpts.m` and `loadTEMpts.m` are not called by `OfflineCorrelationGUI.m`. They are
kept only so this directory is a faithful snapshot of what was retired.

## Golden data

Inputs and expected outputs live in `../data/` and `../data/TestResults/`
(`TransformedData.csv`, `OverlayFLMTEM.tif`).

The pre-refactor state of the whole repository is tagged `legacy-matlab-baseline`.
