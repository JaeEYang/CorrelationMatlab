# CorrelationMatlab

A cryo-CLEM correlation workflow, moving from MATLAB to a napari application with
SerialEM integration.

- **`napari-offilinecorrelation-test/`** — the napari plugin (active development).
  See [environment.yml](./environment.yml) for the development environment.
- **`matlab/`** — the original MATLAB tool, now a
  [frozen reference implementation](./matlab/README.md). It is being retired and
  is kept only to define golden outputs for the Python port.

## MATLAB reference tool (frozen)

### Requirements
- MATLAB R2020a or newer
- Image Processing Toolbox
- (optional) Fiji/ImageJ if using with coordinate generation and correlation plugins

### Installation
1. Install required software
2. Run MATLAB main script [matlab/OfflineCorrelationGUI.m](./matlab/OfflineCorrelationGUI.m)
3. Use the test data here:
   - [data/](./data/)
   - [Sample FLM](./data/X7Y6_FLM_RGB_2.tif)
   - [Sample TEM](./data/TEM_square6_470x.tif)
   - [Sample FLM registration](./data/Item2_X7Y6_FLM_RegSpread9.csv)
   - [Sample TEM registration](./data/Item1_ER80_G3_470x_Pt6_TEM_RegSpread9.csv)
