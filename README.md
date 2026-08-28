# CorrelationMatlab

A cryo-CLEM correlation workflow for registering fluorescence light microscopy
(FLM) images with transmission electron microscopy (TEM) images and exporting
correlated targets to SerialEM.

The napari plugin is the active implementation. The original MATLAB application
is frozen and retained only as a behavioral reference for the Python port.

## Active napari application

### Requirements

- Miniforge, Miniconda, or another conda-compatible environment manager
- A graphical environment capable of running napari

### Development setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate correlation-napari
python -m pip install -e "napari-offilinecorrelation-test[test]"
```

The environment file pins the development stack used by this project, including
Python 3.11, napari 0.6.4, and PyQt5.

### Launch

```bash
napari
```

In napari's **Plugins** menu, open one of the two Correlation 2D-3D widgets:

- **Offline Correlation** loads images and registration points and provides
  image orientation controls.
- **SerialEM Integration** loads Navigator files, displays associated maps,
  performs FLM-to-TEM registration, and exports queued Navigator points.

Sample images, registration points, Navigator files, and reference outputs are
available in [`data/`](./data/).

### Tests

After installing the editable package with the `test` extra:

```bash
python -m pytest napari-offilinecorrelation-test/tests
```

## Repository layout

- **[`napari-offilinecorrelation-test/`](./napari-offilinecorrelation-test/)** —
  active napari plugin development.
- **[`matlab/`](./matlab/)** — frozen MATLAB reference implementation used to
  define golden outputs for the Python port.
- **[`data/`](./data/)** — sample inputs, Navigator files, and MATLAB reference
  outputs.
- **[`environment.yml`](./environment.yml)** — reproducible development
  environment.

## MATLAB reference tool (frozen)

The MATLAB application is not used by the Python plugin. Run it only when
comparing Python behavior against the legacy reference implementation. See
[`matlab/README.md`](./matlab/README.md) for its role and conventions.

### Requirements

- MATLAB R2020a or newer
- Image Processing Toolbox
- Optional: Fiji/ImageJ when using coordinate-generation and correlation plugins

### Running the reference tool

1. Install the required MATLAB software.
2. Run [matlab/OfflineCorrelationGUI.m](./matlab/OfflineCorrelationGUI.m).
3. Use the sample files in [`data/`](./data/), including:
   - [FLM image](./data/X7Y6_FLM_RGB_2.tif)
   - [TEM image](./data/TEM_square6_470x.tif)
   - [FLM registration points](./data/Item2_X7Y6_FLM_RegSpread9.csv)
   - [TEM registration points](./data/Item1_ER80_G3_470x_Pt6_TEM_RegSpread9.csv)
