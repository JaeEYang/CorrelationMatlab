import numpy as np
from pathlib import Path
from magicgui.widgets import FileEdit, PushButton, ComboBox, Container, Label, Slider
import mrcfile
from skimage import io
from napari.layers import Image
from scipy.ndimage import rotate as ndi_rotate

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
def parse_nav(nav_path):
    """Parse SerialEM .nav file for montage maps and points."""
    try:
        nav_path = Path(nav_path)
    except Exception:
        return {}, []

    if not nav_path.exists():
        return {}, []

    maps, points = {}, []
    current_map, map_id = None, None

    try:
        with open(nav_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "Map":
                    current_map, map_id = {}, None
                elif line.startswith("MapID"):
                    map_id = int(line.split("=")[1].strip())
                    maps[map_id] = current_map
                elif line.startswith("ImageFile") and current_map is not None:
                    current_map["file"] = line.split("=")[1].strip()
                elif line.startswith("PieceCoordinates") and current_map is not None:
                    coords = line.split("=")[1].strip().split()
                    current_map.setdefault("coords", []).append((int(coords[0]), int(coords[1])))
                elif line.startswith("Point"):
                    points.append({})
                elif line.startswith("StageX") and points:
                    points[-1]["x"] = float(line.split("=")[1].strip())
                elif line.startswith("StageY") and points:
                    points[-1]["y"] = float(line.split("=")[1].strip())
                elif line.startswith("StageZ") and points:
                    points[-1]["z"] = float(line.split("=")[1].strip())
    except Exception:
        return {}, []

    return maps, points


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
    if points is None or len(points) == 0:
        return points
    h, w = shape[:2]
    cy, cx = h / 2, w / 2
    theta = -np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = points.copy().astype(float)
    ys, xs = pts[:, 0], pts[:, 1]
    coords = np.stack([xs - cx, ys - cy], axis=1)
    rotated = coords @ R.T
    return np.stack([rotated[:, 1] + cy, rotated[:, 0] + cx], axis=1)


# ---------------------------
# Load Images Widget
# ---------------------------
def load_images_widget(viewer: "napari.viewer.Viewer") -> Container:
    mrc_edit = FileEdit(label="", mode="r", filter="*.mrc *.st *.tif *.tiff *.png *.jpg")
    nav_edit = FileEdit(label="", mode="r", filter="*.nav")
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
        if layer is None: return
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
        new_pts = points.copy(); new_pts[:, 0] = img_shape[0] - 1 - points[:, 0]; return new_pts
    def flip_horizontal(points, img_shape):
        new_pts = points.copy(); new_pts[:, 1] = img_shape[1] - 1 - points[:, 1]; return new_pts

    def transform_image(img_fn, pts_fn=None):
        layer = assigned_images.get(name)
        if layer is None: return
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
        layer = viewer.add_points(np.empty((0, 2)), name=f"{name} Points", size=12,
                                  face_color="red" if name == "Image 1" else "blue", edge_color="black")
        assigned_points[name] = layer
        original_points[name] = layer.data.copy()

    new_pts_btn.clicked.connect(on_new_points)
    return Container(widgets=[combo, label, clear_btn, angle_slider, flipv_btn, fliph_btn, new_pts_btn])


# ---------------------------
# Load Points Widget
# ---------------------------
def load_points_widget(viewer: "napari.viewer.Viewer") -> Container:
    file_edit, combo, button = FileEdit(label="", mode="r", filter="*.csv"), ComboBox(label="Assign to", choices=["Image 1", "Image 2"]), PushButton(text="Load Points")

    def _on_click(event=None):
        path = file_edit.value
        if not path or not Path(path).exists():
            return
        pts = np.loadtxt(str(path), delimiter=",")
        if pts.ndim == 1: pts = pts.reshape(1, -1)
        if pts.shape[1] >= 2:
            which = combo.value
            y0, x0 = offsets.get(which, (0, 0))
            pts[:, 0] += y0; pts[:, 1] += x0
            layer = viewer.add_points(pts[:, :2], name=f"{which} Points", size=12,
                                      face_color="red" if which == "Image 1" else "blue", edge_color="black")
            assigned_points[which] = layer
            original_points[which] = layer.data.copy()

    button.clicked.connect(_on_click)
    return Container(widgets=[file_edit, combo, button])
