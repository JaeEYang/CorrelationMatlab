import numpy as np
from pathlib import Path
from magicgui.widgets import FileEdit, PushButton, ComboBox, Container, Label, Slider
import mrcfile
from skimage import io
from napari.layers import Image
from scipy.ndimage import rotate as ndi_rotate
from correlation2d3d import NavBuilt as nb
from correlation2d3d import SupportNav as sn
from qtpy.QtWidgets import QFileDialog
from napari.viewer import Viewer
import re
from typing import List, Tuple, Optional
import os

# ---------------------------
# Global state
# ---------------------------
assigned_images = {"Image 1": None, "Image 2": None}
assigned_points = {"Image 1": None, "Image 2": None}
original_images = {}
original_points = {"Image 1": None, "Image 2": None}
offsets = {"Image 1": (0, 0), "Image 2": (0, 0)}


# ---------------------------
# NAV parsing
# ---------------------------
def get_last_item_number(nav_path: str) -> int:
    last_num = 0
    pattern = re.compile(r"\[Item\s*=\s*(\d+)\]")
    with open(nav_path, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                num = int(match.group(1))
                last_num = max(last_num, num)
    return last_num

def _normalize_windows_path(s: str) -> str:
    # strip quotes and normalize slashes
    s = s.strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    return s

def _unique_existing_paths(paths: List[Path]) -> List[Path]:
    out = []
    seen = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
        elif key not in seen and not p.exists():
            # keep non-existing too (search roots), but ensure uniqueness
            seen.add(key)
            out.append(p)
    return out

def _case_insensitive_rglob(root: Path, name: str):
    target_lower = name.lower()
    for p in root.rglob("*"):
        if p.name.lower() == target_lower and p.is_file():
            yield p

def _resolve_map_path(
    nav_path: str,
    mapfile_field: str,
    extra_root: Optional[Path] = None,
    prefix_maps: Optional[List[Tuple[str, Path]]] = None,
) -> Optional[Path]:
   
    nav_dir = Path(nav_path).parent

    # --- candidate roots to search (order matters)
    candidate_roots: List[Path] = []
    if extra_root:
        candidate_roots.append(Path(extra_root))
        candidate_roots.append(Path(extra_root) / "data")
    candidate_roots.append(nav_dir)
    candidate_roots.append(nav_dir / "data")

    candidate_roots = _unique_existing_paths(candidate_roots)

    # --- 0) Normalize the incoming field
    mf_str = _normalize_windows_path(mapfile_field)
    mf_path = Path(mf_str)

    # --- 1) Exact path as-given
    if mf_path.is_file():
        print(f"ℹ Resolved by exact normalized path: {mf_path}")
        return mf_path

    # --- 2) Treat as relative to NAV directory
    rel_cand = (nav_dir / mf_path).resolve()
    if rel_cand.is_file():
        print(f"ℹ Resolved as path relative to NAV dir: {rel_cand}")
        return rel_cand

    # --- 3) Apply prefix remaps (Windows → local)
    # Build default prefix maps if user didn't pass any.
    # You can add more tuples if you have other drive letters or UNC shares.
    default_maps: List[Tuple[str, Path]] = []
    if extra_root:
        # Common case: map Windows drive prefix to repo/data
        default_maps.extend([
            ("X:/RawData/wright/jyang525", Path(extra_root) / "data"),
            ("X:/RawData", Path(extra_root) / "data"),
        ])
    # Accept caller-provided maps and place them before defaults.
    prefix_maps = (prefix_maps or []) + default_maps

    # Try each prefix map
    for win_prefix, local_root in prefix_maps:
        norm_prefix = _normalize_windows_path(win_prefix).rstrip("/")
        if mf_str.lower().startswith(norm_prefix.lower() + "/"):
            tail = mf_str[len(norm_prefix) + 1 :]  # path under the prefix
            remapped = (Path(local_root) / tail).resolve()
            if remapped.is_file():
                print(f"ℹ Resolved via prefix map [{win_prefix} -> {local_root}]: {remapped}")
                return remapped
            # also try just the filename under that root
            fname = Path(tail).name
            direct = (Path(local_root) / fname).resolve()
            if direct.is_file():
                print(f"ℹ Resolved via prefix map (filename-only) [{win_prefix} -> {local_root}]: {direct}")
                return direct

    # --- 4) Case-insensitive filename search across candidate roots
    target_name = Path(mf_str).name
    for root in candidate_roots:
        for hit in _case_insensitive_rglob(root, target_name):
            print(f"ℹ Resolved by case-insensitive filename search under {root}: {hit}")
            return hit

    # --- 5) Fallback by stem + common extensions
    stem = Path(target_name).stem
    exts = [".st", ".mrc", ".mrcs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    for root in candidate_roots:
        for ext in exts:
            # exact stem match
            for p in root.rglob(stem + ext):
                if p.is_file():
                    print(f"ℹ Resolved by stem+ext under {root}: {p}")
                    return p
            # case-insensitive stem match
            for p in root.rglob(f"*{ext}"):
                if p.is_file() and Path(p).stem.lower() == stem.lower():
                    print(f"ℹ Resolved by case-insensitive stem+ext under {root}: {p}")
                    return p

    print(f"⚠ Could not resolve {target_name} under {[str(r) for r in candidate_roots]}")
    return None


def _read_map_array(map_path: Path) -> np.ndarray:
    """
    Read image/volume from disk. Use mrcfile for .st/.mrc/.mrcs, otherwise skimage.io.imread.
    """
    suf = map_path.suffix.lower()
    if suf in {".st", ".mrc", ".mrcs"}:
        with mrcfile.open(str(map_path), permissive=True) as mrc:
            data = mrc.data  # could be 2D or 3D
        return np.asarray(data)
    else:
        return io.imread(str(map_path))

def points2nav_widget(viewer: "Viewer") -> Container:
    # GUI widgets
    csv_edit = FileEdit(label="Points CSV", mode="r", filter="*.csv")
    nav_edit = FileEdit(label="Template NAV", mode="r", filter="*.nav")
    combo = ComboBox(label="Assign to Map", choices=[])

    btn_view = PushButton(text="View Points")
    btn_add = PushButton(text="Add Points to NAV")
    btn_show_map = PushButton(text="Show Map")

    # --- Step 1: Load CSV and preview points
    def _on_view(event=None):
        csv_path = csv_edit.value
        if not csv_path or not Path(csv_path).exists():
            print(" CSV file not found")
            return

        coords = np.loadtxt(str(csv_path), delimiter=",")
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        if coords.shape[1] < 2:
            print(" CSV must have at least 2 columns (X,Y)")
            return

        # Preview only XY for Napari
        pts = coords[:, :2]
        if "Preview Points" in viewer.layers:
            viewer.layers["Preview Points"].data = pts
        else:
            viewer.add_points(
                pts,
                name="Preview Points",
                size=10,
                face_color="yellow",
            )
        print(f" Displayed {pts.shape[0]} points from {csv_path}")

    # --- Step 2: Load NAV file (after CSV is already loaded)
    def _on_nav_change(event=None):
        nav_path = nav_edit.value
        if nav_path and Path(nav_path).exists():
            navdata = nb.parseNavFile(str(nav_path))
            if navdata.Maps:
                combo._maps = navdata.Maps  # store objects
                combo.choices = [
                    f"Map {m.Label} (ID={m.MapID}, Regis={m.Regis}, File={Path(m.MapFile).name if m.MapFile else 'None'})"
                    for m in navdata.Maps
                ]
                print(f" Loaded {len(navdata.Maps)} map(s) from {nav_path}")

    nav_edit.changed.connect(_on_nav_change)

    # --- Step 3: Add points into NAV
    def _on_add(event=None):
        csv_path = csv_edit.value
        nav_path = nav_edit.value

        if not csv_path or not Path(csv_path).exists():
            print(" CSV file not found")
            return
        if not nav_path or not Path(nav_path).exists():
            print(" Template NAV file not found")
            return
        if not hasattr(combo, "_maps") or combo.value is None:
            print(" Please select a map from the dropdown")
            return

        # Ask where to save
        out_path, _ = QFileDialog.getSaveFileName(
            None, "Save Output NAV", "output.nav", "NAV Files (*.nav)"
        )
        if not out_path:
            print("⚠ Save cancelled")
            return

        # Parse NAV and chosen map
        navdata = nb.parseNavFile(str(nav_path))

        # Find the index of the currently selected item
        if combo.value not in combo.choices:
            print("No valid map selected")
            return
        
        map_index = combo.choices.index(combo.value)
        map_item = combo._maps[map_index]
        
        try:
            coords = viewer.layers["Preview Points"].data
        except KeyError:
            print("⚠ No Preview Points layer found")
            return
        
        # Load CSV coords
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        # --- FIX: get last item number once from template
        last_item_num = get_last_item_number(nav_path)
        print(f"Last item number found in template NAV = {last_item_num}")

        new_points = []

        for offset, row in enumerate(coords, start=1):
            if len(row) == 2:
                x, y = row
                z = 0.0
            else:
                x, y, z = row[:3]

            item_num = last_item_num + offset  # stable numbering

            p = nb.PointItem()
            p.Label = str(item_num)
            p.StageXYZ = [float(x), float(y), float(z)]
            p.PtsX = float(x)
            p.PtsY = float(y)
            p.DrawnID = map_item.MapID
            p.Regis = map_item.Regis
            new_points.append((item_num, p))

        # Write out new NAV
        with open(out_path, "w") as f:
            with open(nav_path, "r") as fin:
                f.write(fin.read()) # preserve everything

            for item_num, p in new_points:
                lines = p.getText()[:]
                if lines:
                    if lines[0].startswith("[Item"):
                        lines[0] = f"[Item = {item_num}]"
                    else:
                        lines.insert(0, f"[Item = {item_num}]")

                f.write("\n\n")
                f.write("\n".join(lines))

        print(f" NAV written: {out_path}")

    # Hook up buttons
    btn_view.clicked.connect(_on_view)
    btn_add.clicked.connect(_on_add)

    #return Container(widgets=[csv_edit, btn_view, nav_edit, combo, btn_add])

     # --- Step 4: Show map image from NAV
    def _on_show_map(event=None):
        nav_path = nav_edit.value   
        if not nav_path or not Path(nav_path).exists():
            print("⚠ Template NAV file not found")
            return
        if not hasattr(combo, "_maps") or combo.value is None:
            print("⚠ No map selected")
            return

        navdata = nb.parseNavFile(str(nav_path))
        map_index = combo.choices.index(combo.value)
        map_item = combo._maps[map_index]

        print(f"DEBUG: map_item.MapFile={map_item.MapFile}")
        print(f"DEBUG: map_item.Label={map_item.Label}")

        if not map_item.MapFile:
            print(f"⚠ Selected map {map_item.Label} has no MapFile")
            return

        # Extract filename only
        #map_filename = Path(map_item.MapFile).name

        # Resolve path near NAV
        map_path = _resolve_map_path(
            nav_path,
            map_item.MapFile,
            extra_root=Path("/Users/jyang525/Documents/MATLAB/CorRelator/ER80_G3_TestingInput_3/SmallModuleDevelopment/github_CorrelationMatlab/"),
            prefix_maps=[
                ("X:/RawData/wright/jyang525", Path("/Users/jyang525/Documents/MATLAB/CorRelator/.../github_CorrelationMatlab/data")),
                 # add more if needed
            ],
        )

        map_path = _resolve_map_path(nav_path, map_item.MapFile, extra_root=Path("/Users/jyang525/Documents/MATLAB/CorRelator/ER80_G3_TestingInput_3/SmallModuleDevelopment/github_CorrelationMatlab/"))
        if not map_path:
            print(f"⚠ Could not locate {Path(map_item.MapFile).name} near {nav_path}")
            return
        if map_path.is_dir():
            print(f"⚠ Resolved map_path is a directory, not a file: {map_path}")
            return
        
        print(f"ℹ Using map file: {map_path}")

        try:
            arr = _read_map_array(map_path)
            viewer.add_image(arr, name=f"Map {map_item.Label}")
            viewer.reset_view()
            print(f"✅ Loaded map {map_item.Label} from {map_path} | shape={arr.shape}")
        except Exception as e:
            print(f"⚠ Failed to read map image from {map_path}: {e}")



    btn_show_map.clicked.connect(_on_show_map)

    # Return container with all buttons
    return Container(
        widgets=[csv_edit, btn_view, nav_edit, combo, btn_add, btn_show_map]
    )

def reconstruct_from_nav(mrc_path: Path, coords):
    """Reconstruct montage from NAV coords."""
    with mrcfile.open(str(mrc_path), permissive=True) as mrc:
        tiles = np.copy(mrc.data)

    if tiles.ndim < 3:
        raise ValueError("Expected stack of 2D tiles in MRC/ST")

    tile_h, tile_w = tiles.shape[1:3]
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    min_x, min_y = min(xs), min(ys)
    xs, ys = [x - min_x for x in xs], [y - min_y for y in ys]

    canvas_h, canvas_w = max(ys) + tile_h, max(xs) + tile_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=tiles.dtype)

    for i, (x, y) in enumerate(zip(xs, ys)):
        tile = tiles[i]
        h, w = tile.shape
        canvas[y:y+h, x:x+w] = tile

    return canvas


# ---------------------------
# MDOC parsing
# ---------------------------
def parse_mdoc(mdoc_path: Path):
    """Parse .mdoc file, returning coords, full montage size and per-tile size."""
    if not Path(mdoc_path).exists():
        return {"coords": [], "full_size": None, "image_size": None}

    coords, full_size, image_size = [], None, None
    current_z, pending_xy = None, {"aligned": None, "raw": None}

    def flush_pending():
        if pending_xy["aligned"] is not None:
            coords.append(tuple(pending_xy["aligned"]))
        elif pending_xy["raw"] is not None:
            coords.append(tuple(pending_xy["raw"]))

    with open(mdoc_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("[ZValue"):
                try:
                    # Example line: "[ZValue = 0]"
                    val = line.split("=")[1].strip().strip("]")
                    current_z = int(val)
                except Exception:
                    current_z = None
                if current_z is not None:
                    if 'pending_xy' in locals():
                        flush_pending()
                    pending_xy = {"aligned": None, "raw": None}

            if line.startswith("AlignedPieceCoords"):
                parts = line.split("=", 1)[1].strip().split()
                if len(parts) >= 2:
                    pending_xy["aligned"] = (int(float(parts[0])), int(float(parts[1])))
                continue

            if line.startswith("PieceCoordinates"):
                parts = line.split("=", 1)[1].strip().split()
                if len(parts) >= 2:
                    pending_xy["raw"] = (int(float(parts[0])), int(float(parts[1])))
                continue

            if line.startswith("ImageSize"):
                parts = line.split("=", 1)[1].strip().split()
                if len(parts) >= 2:
                    image_size = (int(parts[0]), int(parts[1]))
                continue

            if line.startswith("FullMontSize"):
                parts = line.split("=", 1)[1].strip().split()
                if len(parts) >= 2:
                    full_size = (int(parts[0]), int(parts[1]))
                continue

        if current_z is not None:
            flush_pending()

    return {"coords": coords, "full_size": full_size, "image_size": image_size}


def reconstruct_mdoc_montage(st_path: Path, mdoc_path: Path):
    """Reconstruct montage from .st stack + .mdoc coords with clipping."""
    meta = parse_mdoc(mdoc_path)
    coords = meta["coords"]
    if not coords:
        raise ValueError("No coordinates in MDOC")

    with mrcfile.open(str(st_path), permissive=True) as mrc:
        tiles = np.copy(mrc.data)

    tile_h, tile_w = tiles.shape[1:3]
    nch = None if tiles.ndim == 3 else tiles.shape[3]

    if meta["full_size"] is not None:
        canvas_w, canvas_h = meta["full_size"]
    else:
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        min_x, min_y = min(xs), min(ys)
        xs, ys = [x - min_x for x in xs], [y - min_y for y in ys]
        canvas_w, canvas_h = max(xs) + tile_w, max(ys) + tile_h

    canvas = np.zeros((canvas_h, canvas_w) + (() if nch is None else (nch,)), dtype=tiles.dtype)

    n = min(len(coords), tiles.shape[0])
    for i in range(n):
        x, y = coords[i]
        tile = tiles[i]
        h, w = tile.shape[:2]

        # Clip start and end
        x_start, y_start = max(x, 0), max(y, 0)
        x_end, y_end = min(x + w, canvas_w), min(y + h, canvas_h)

        tile_x_start = 0 if x >= 0 else -x
        tile_y_start = 0 if y >= 0 else -y
        tile_x_end = tile_x_start + (x_end - x_start)
        tile_y_end = tile_y_start + (y_end - y_start)

        if x_end <= x_start or y_end <= y_start:
            print(f"⚠️ Skipping tile {i}: completely out of bounds at ({x},{y})")
            continue

        if nch is None:
            canvas[y_start:y_end, x_start:x_end] = tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        else:
            canvas[y_start:y_end, x_start:x_end, :] = tile[tile_y_start:tile_y_end,
                                                           tile_x_start:tile_x_end, :]

    return canvas


# ---------------------------
# Rotation helpers
# ---------------------------
def prepare_canvas(data):
    h, w = data.shape[:2]
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    cy, cx = diag // 2, diag // 2
    canvas = np.zeros((diag, diag) + (() if data.ndim == 2 else (data.shape[2],)), dtype=data.dtype)
    y0, x0 = cy - h // 2, cx - w // 2
    canvas[y0:y0+h, x0:x0+w] = data
    return canvas, (y0, x0)


def rotate_image_fixed(data, angle):
    return ndi_rotate(data, angle, reshape=False, order=1, mode="constant", cval=0)


def rotate_points_fixed(points, angle, shape):
    if points is None:
        return points
    pts = np.asarray(points)
    if pts.size == 0:
        return pts
    h, w = shape[:2]
    cy, cx = h / 2.0, w / 2.0
    theta = -np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    ys, xs = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    coords = np.stack([xs - cx, ys - cy], axis=1)
    rotated = coords @ R.T
    out = np.stack([rotated[:, 1] + cy, rotated[:, 0] + cx], axis=1).astype(np.float32)
    return out


# ---------------------------
# Load Images Widget
# ---------------------------
def load_images_widget(viewer: "napari.viewer.Viewer") -> Container:
    mrc_edit = FileEdit(label="Image File", mode="r", filter="*.mrc *.st *.tif *.tiff *.png *.jpg")
    nav_edit = FileEdit(label="Navigator file(*.nav)", mode="r", filter="*.nav")
    button = PushButton(text="Load Images")

    # toggle: decide if we also show raw stack (default: False to avoid confusion)
    show_raw = False

    def _on_click(event=None):
        mrc_path = mrc_edit.value
        nav_path = nav_edit.value

        if not mrc_path or not Path(mrc_path).exists():
            print("❌ Please select an MRC/ST or image file")
            return

        ext = Path(mrc_path).suffix.lower()
        name = Path(mrc_path).stem

        # --- MRC/ST branch ---
        if ext in [".mrc", ".st"]:
            try:
                with mrcfile.open(str(mrc_path), permissive=True) as mrc:
                    raw = np.copy(mrc.data)
            except Exception as e:
                print(f"⚠️ Failed to open stack: {e}")
                return

            # try .mdoc montage
            mdoc_path = Path(str(mrc_path) + ".mdoc")
            montage = None
            if mdoc_path.exists():
                try:
                    montage = reconstruct_mdoc_montage(Path(mrc_path), mdoc_path)
                except Exception as e:
                    print(f"⚠️ Mdoc failed: {e}")

            if montage is not None:
                canvas, offset = prepare_canvas(montage)
                viewer.add_image(canvas, name=f"{name} [montage]", colormap="gray")
                offsets["Image 1"] = offset
                if show_raw:
                    viewer.add_image(raw, name=f"{name} [raw stack]", colormap="gray")
            else:
                # fallback: show raw stack
                viewer.add_image(raw, name=f"{name} [raw stack]", colormap="gray")

                # optional: also show central slice as quick preview
                if raw.ndim == 3:
                    mid = raw.shape[0] // 2
                    canvas, offset = prepare_canvas(raw[mid])
                    viewer.add_image(canvas, name=f"{name} [preview]", colormap="gray")
                    offsets["Image 1"] = offset
                elif raw.ndim == 2:
                    canvas, offset = prepare_canvas(raw)
                    viewer.add_image(canvas, name=f"{name} [preview]", colormap="gray")
                    offsets["Image 1"] = offset

        # --- Normal images ---
        else:
            img = io.imread(str(mrc_path))
            canvas, offset = prepare_canvas(img)
            if img.ndim == 2:
                viewer.add_image(canvas, name=f"{name} [image]", colormap="gray")
            else:
                viewer.add_image(canvas, name=f"{name} [image]")
            offsets["Image 1"] = offset

        # --- Optional NAV parsing (legacy) ---
        if nav_path and Path(nav_path).exists():
            maps, _ = parse_nav(nav_path)
            if maps:
                for mid, info in maps.items():
                    try:
                        mrc_file = Path(nav_path).parent / info["file"]
                        if not mrc_file.exists():
                            continue
                        mosaic = reconstruct_from_nav(mrc_file, info["coords"])
                        mosaic_canvas, _ = prepare_canvas(mosaic)
                        viewer.add_image(mosaic_canvas, name=f"Montage Map {mid}", colormap="gray")
                    except Exception as e:
                        print(f"⚠️ Failed NAV montage {mid}: {e}")

    button.clicked.connect(_on_click)
    return Container(widgets=[mrc_edit, nav_edit, button])


# ---------------------------
# Image Panel
# ---------------------------
def make_image_panel(viewer, name: str = "Image 1") -> Container:
    combo = ComboBox(label=f"Select {name}", choices=lambda *a: [l.name for l in viewer.layers if isinstance(l, Image)])
    label = Label(value=f"{name}: None")
    clear_btn, angle_slider = PushButton(text="Clear"), Slider(min=0, max=360, step=1, value=0, label="Rotate °")
    flipv_btn, fliph_btn, new_pts_btn = PushButton(text="Flip V"), PushButton(text="Flip H"), PushButton(text="New Points Layer")

    def on_select(event=None):
        if combo.value:
            assigned_images[name] = viewer.layers[combo.value]
            label.value, original_images[name] = f"{name}: {combo.value}", np.copy(assigned_images[name].data)

    def on_clear(event=None):
        assigned_images[name] = assigned_points[name] = None
        original_images.pop(name, None)
        original_points.pop(name, None)
        label.value = f"{name}: None"

    def on_angle_change(event=None):
        layer = assigned_images.get(name)
        if layer is None:
            return
        base = original_images.get(name, layer.data)
        if base.ndim == 2 or (base.ndim == 3 and base.shape[-1] in (3, 4)):
            angle = angle_slider.value
            rotated = rotate_image_fixed(base, angle)
            layer.data = rotated
            pts_layer = assigned_points.get(name)
            if pts_layer is not None:
                base_points = original_points.get(name, pts_layer.data)
                pts_layer.data = rotate_points_fixed(base_points, angle, base.shape)

    def flip_vertical(points, img_shape):
        new_pts = points.copy()
        new_pts[:, 0] = img_shape[0] - 1 - points[:, 0]
        return new_pts

    def flip_horizontal(points, img_shape):
        new_pts = points.copy()
        new_pts[:, 1] = img_shape[1] - 1 - points[:, 1]
        return new_pts

    def transform_image(img_fn, pts_fn=None):
        layer = assigned_images.get(name)
        if layer is None:
            return
        data = layer.data
        if data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in (3, 4)):
            layer.data = img_fn(data)
            pts_layer = assigned_points.get(name)
            if pts_layer is not None and pts_fn is not None:
                pts_layer.data = pts_fn(pts_layer.data, data.shape)

    combo.changed.connect(on_select)
    clear_btn.clicked.connect(on_clear)
    angle_slider.changed.connect(on_angle_change)
    flipv_btn.clicked.connect(lambda e: transform_image(np.flipud, flip_vertical))
    fliph_btn.clicked.connect(lambda e: transform_image(np.fliplr, flip_horizontal))

    def on_new_points(event=None):
        layer = viewer.add_points(
            np.empty((0, 2)),
            name=f"{name} Points",
            size=12,
            face_color="red" if name == "Image 1" else "blue",
        )
        assigned_points[name] = layer
        original_points[name] = layer.data.copy()

    new_pts_btn.clicked.connect(on_new_points)
    return Container(widgets=[combo, label, clear_btn, angle_slider, flipv_btn, fliph_btn, new_pts_btn])


# ---------------------------
# Load Points Widget (robust CSV)
# ---------------------------
def load_points_widget(viewer: "napari.viewer.Viewer") -> Container:
    file_edit = FileEdit(label="Coordinates file (*.csv)", mode="r", filter="*.csv")
    combo = ComboBox(label="Assign to", choices=["Image 1", "Image 2"])
    button = PushButton(text="Load Points")

    def _read_csv_points(path: Path) -> np.ndarray:
        """
        Robust CSV reader:
        - accepts header or not
        - delimiters: comma/semicolon/space/tab
        - columns x,y or y,x (auto by header or heuristic)
        - returns Nx2 in (y, x) as float32
        """
        import csv

        with open(path, "r", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=",; \t")
            except Exception:
                class _D:
                    delimiter = ","
                dialect = _D()

            reader = csv.reader(f, dialect)
            rows = [r for r in reader if len(r) > 0]

        if not rows:
            raise ValueError("Empty CSV.")

        has_header = any(any(c.isalpha() for c in cell) for cell in rows[0])
        data_rows = rows[1:] if has_header else rows

        numeric = []
        for r in data_rows:
            vals = []
            for cell in r:
                cell = cell.strip()
                if cell == "":
                    continue
                cell = cell.replace(",", ".") if getattr(dialect, "delimiter", ",") != "," else cell
                try:
                    vals.append(float(cell))
                except Exception:
                    continue
            if len(vals) >= 2:
                numeric.append(vals[:3])  # allow optional Z, will be dropped

        if not numeric:
            raise ValueError("No numeric rows with at least 2 values were found.")

        arr = np.asarray(numeric, dtype=np.float32)

        # By header
        if has_header:
            hdr = [h.strip().lower() for h in rows[0]]
            try:
                ix = hdr.index("x")
                iy = hdr.index("y")
                yx = np.stack([arr[:, iy], arr[:, ix]], axis=1)
                return yx.astype(np.float32)
            except ValueError:
                pass

        # Heuristic
        if arr.shape[1] >= 2:
            x_like, y_like = arr[:, 0], arr[:, 1]
            if np.nanvar(y_like) >= np.nanvar(x_like):
                yx = np.stack([arr[:, 1], arr[:, 0]], axis=1)
                return yx.astype(np.float32)
            else:
                return arr[:, :2].astype(np.float32)

        raise ValueError("CSV could not be interpreted as 2D points.")

    def _on_click(event=None):
        path = file_edit.value
        if not path or not Path(path).exists():
            print("❌ Select a CSV first.")
            return

        try:
            pts_yx = _read_csv_points(Path(path))  # (N,2) as (y,x)
            which = combo.value or "Image 1"
            y0, x0 = offsets.get(which, (0, 0))
            pts_yx = pts_yx.copy()
            pts_yx[:, 0] += float(y0)
            pts_yx[:, 1] += float(x0)

            layer = viewer.add_points(
                pts_yx,
                name=f"{which} Points",
                size=12,
                face_color="red" if which == "Image 1" else "blue",
            )
            assigned_points[which] = layer
            original_points[which] = layer.data.copy()
            print(f"✅ Loaded {pts_yx.shape[0]} points into '{which}'.")

        except Exception as e:
            print(f"❌ Failed to load points: {e}")

    button.clicked.connect(_on_click)
    return Container(widgets=[file_edit, combo, button])
