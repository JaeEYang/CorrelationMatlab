import numpy as np
from pathlib import Path
from magicgui.widgets import FileEdit, PushButton, ComboBox, Container, Label
import mrcfile
from skimage import io
from napari.layers import Image

# Global state: linked images and points
assigned_images = {"Image 1": None, "Image 2": None}
assigned_points = {"Image 1": None, "Image 2": None}


# ---------------------------
# NAV parsing + reconstruction
# ---------------------------
def parse_nav(nav_path):
    """Parse SerialEM .nav file for montage maps and points."""
    try:
        nav_path = Path(nav_path)
    except Exception:
        print(f"⚠️ Invalid nav_path: {nav_path}")
        return {}, []

    if not nav_path.exists():
        print(f"⚠️ NAV file does not exist: {nav_path}")
        return {}, []

    maps = {}
    points = []
    current_map = None
    map_id = None

    try:
        with open(nav_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "Map":
                    current_map = {}
                    map_id = None
                elif line.startswith("MapID"):
                    map_id = int(line.split("=")[1].strip())
                    maps[map_id] = current_map
                elif line.startswith("ImageFile"):
                    if current_map is not None:
                        current_map["file"] = line.split("=")[1].strip()
                elif line.startswith("PieceCoordinates"):
                    coords = line.split("=")[1].strip().split()
                    if current_map is not None:
                        if "coords" not in current_map:
                            current_map["coords"] = []
                        current_map["coords"].append((int(coords[0]), int(coords[1])))
                elif line.startswith("Point"):
                    points.append({})
                elif line.startswith("StageX") and points:
                    points[-1]["x"] = float(line.split("=")[1].strip())
                elif line.startswith("StageY") and points:
                    points[-1]["y"] = float(line.split("=")[1].strip())
                elif line.startswith("StageZ") and points:
                    points[-1]["z"] = float(line.split("=")[1].strip())
    except Exception as e:
        print(f"❌ Failed to parse NAV file {nav_path}: {e}")
        return {}, []

    return maps, points


def reconstruct_from_nav(mrc_path: Path, coords):
    """Reconstruct montage image from tile stack + piece coordinates."""
    with mrcfile.open(str(mrc_path), permissive=True) as mrc:
        tiles = np.copy(mrc.data)

    if tiles.ndim != 3:
        raise ValueError("Expected stack of 2D tiles in MRC")

    h, w = tiles.shape[1:]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    min_x, min_y = min(xs), min(ys)
    xs = [x - min_x for x in xs]
    ys = [y - min_y for y in ys]

    canvas_h = max(ys) + h
    canvas_w = max(xs) + w
    canvas = np.zeros((canvas_h, canvas_w), dtype=tiles.dtype)

    for i, (x, y) in enumerate(zip(xs, ys)):
        canvas[y:y+h, x:x+w] = tiles[i]

    return canvas


# ---------------------------
# Load Images Widget
# ---------------------------
def load_images_widget(viewer: "napari.viewer.Viewer") -> Container:
    mrc_edit = FileEdit(label="", mode="r", filter="*.mrc *.tif *.tiff *.png *.jpg")
    nav_edit = FileEdit(label="", mode="r", filter="*.nav")
    button = PushButton(text="Load Images")

    def _on_click(event=None):
        mrc_path = mrc_edit.value
        nav_path = nav_edit.value

        if not mrc_path or not Path(mrc_path).exists():
            print("❌ Please select an MRC or image file")
            return

        # Load MRC or normal image
        if Path(mrc_path).suffix.lower() == ".mrc":
            with mrcfile.open(str(mrc_path), permissive=True) as mrc:
                data = np.copy(mrc.data)
            viewer.add_image(data, name=Path(mrc_path).stem, colormap="gray")
        else:
            img = io.imread(str(mrc_path))
            viewer.add_image(img, name=Path(mrc_path).stem)

        # If nav present, try montage maps
        if nav_path and Path(nav_path).exists():
            maps, _ = parse_nav(nav_path)
            if not maps:
                print(f"⚠️ No valid maps found in {nav_path}")
            else:
                for mid, info in maps.items():
                    mrc_file = Path(nav_path).parent / info["file"]
                    if not mrc_file.exists():
                        continue
                    montage = reconstruct_from_nav(mrc_file, info["coords"])
                    viewer.add_image(montage, name=f"Montage Map {mid}", colormap="gray")

    button.clicked.connect(_on_click)
    return Container(widgets=[mrc_edit, nav_edit, button])


# ---------------------------
# Image Panel (Image 1 / 2)
# ---------------------------
def make_image_panel(viewer, name: str = "Image 1") -> Container:
    combo = ComboBox(
        label=f"Select {name}",
        choices=lambda *args: [layer.name for layer in viewer.layers if isinstance(layer, Image)]
    )
    label = Label(value=f"{name}: None")
    clear_btn = PushButton(text="Clear")
    rotate_btn = PushButton(text="Rotate 90°")
    flipv_btn = PushButton(text="Flip V")
    fliph_btn = PushButton(text="Flip H")
    new_pts_btn = PushButton(text="New Points Layer")

    # ---- Selection handling ----
    def on_select(event=None):
        if combo.value:
            assigned_images[name] = viewer.layers[combo.value]
            label.value = f"{name}: {combo.value}"

    def on_clear(event=None):
        assigned_images[name] = None
        assigned_points[name] = None
        label.value = f"{name}: None"

    # ---- Transform helpers ----
    def rotate_points_90ccw(points, img_shape):
        if len(img_shape) > 2:  # RGB
            H, W = img_shape[:2]
        else:
            H, W = img_shape
        new_pts = np.zeros_like(points)
        # Ajustado para coincidir con np.rot90 (antihorario)
        new_pts[:, 0] = W - 1 - points[:, 1]  # nuevo X = W-1 - Y
        new_pts[:, 1] = points[:, 0]  # nuevo Y = X
        return new_pts

    def flip_vertical(points, img_shape):
        """Vertical flip"""
        if len(img_shape) > 2:
            H, W = img_shape[:2]
        else:
            H, W = img_shape
        new_pts = points.copy()
        new_pts[:, 0] = W - 1 - points[:, 0]  # flip en X
        return new_pts

    def flip_horizontal(points, img_shape):
        """Horizontal flip"""
        if len(img_shape) > 2:
            H, W = img_shape[:2]
        else:
            H, W = img_shape
        new_pts = points.copy()
        new_pts[:, 1] = H - 1 - points[:, 1]  # flip en Y
        return new_pts

    # Apply transform to image + linked points
    def transform_image(img_fn, pts_fn=None):
        layer = assigned_images.get(name)
        if layer is None:
            return
        data = layer.data
        layer.data = img_fn(data)

        points_layer = assigned_points.get(name)
        if points_layer is not None and pts_fn is not None:
            points_layer.data = pts_fn(points_layer.data, data.shape)

    # ---- Button connections ----
    combo.changed.connect(on_select)
    clear_btn.clicked.connect(on_clear)

    rotate_btn.clicked.connect(
        lambda e: transform_image(lambda d: np.rot90(d, k=1), rotate_points_90ccw)
    )
    flipv_btn.clicked.connect(
        lambda e: transform_image(np.flipud, flip_vertical)
    )
    fliph_btn.clicked.connect(
        lambda e: transform_image(np.fliplr, flip_horizontal)
    )

    def on_new_points(event=None):
        layer = viewer.add_points(
            np.empty((0, 2)),
            name=f"{name} Points",
            size=10,
            face_color="red" if name == "Image 1" else "blue"
        )
        assigned_points[name] = layer

    new_pts_btn.clicked.connect(on_new_points)

    return Container(widgets=[
        combo, label, clear_btn,
        rotate_btn, flipv_btn, fliph_btn,
        new_pts_btn
    ])
# ---------------------------
# Load Points Widget
# ---------------------------
def load_points_widget(viewer: "napari.viewer.Viewer") -> Container:
    file_edit = FileEdit(label="", mode="r", filter="*.csv")
    combo = ComboBox(label="Assign to", choices=["Image 1", "Image 2"])
    button = PushButton(text="Load Points")

    def _on_click(event=None):
        path = file_edit.value
        if not path or not Path(path).exists():
            return

        pts = np.loadtxt(str(path), delimiter=",")
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        if pts.shape[1] >= 2:
            layer = viewer.add_points(
                pts[:, :2],
                name=f"{combo.value} Points",
                size=10,
                face_color="red" if combo.value == "Image 1" else "blue"
            )
            assigned_points[combo.value] = layer

    button.clicked.connect(_on_click)
    return Container(widgets=[file_edit, combo, button])
