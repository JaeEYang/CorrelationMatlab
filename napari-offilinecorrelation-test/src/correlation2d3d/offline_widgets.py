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
        print(f"Invalid nav_path: {nav_path}")
        return {}, []

    if not nav_path.exists():
        print(f"NAV file does not exist: {nav_path}")
        return {}, []

    maps = {}
    points = []
    current_map = None

    try:
        with open(nav_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("[Item"):
                    # Start of a new item, could be a map or points
                    current_map = {}
                elif line.startswith("MapID"):
                    if current_map is not None:
                        map_id = int(line.split("=")[1].strip())
                        maps[map_id] = current_map
                elif line.startswith("MapFile"):
                    if current_map is not None:
                        # Extract the filename from the full path
                        full_path = line.split("=")[1].strip()
                        current_map["file"] = Path(full_path).name
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
        print(f"Error: Failed to parse NAV file {nav_path}: {e}")
        return {}, []

    # Filter out any maps that don't have file information
    maps = {k: v for k, v in maps.items() if "file" in v and "coords" in v}
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
# Image Panel (Image 1 / 2)
# ---------------------------
def make_image_panel(viewer: "napari.viewer.Viewer", name: str = "Image 1") -> Container:
    # Common widgets
    combo = ComboBox(label=f"Select {name}", choices=[])
    label = Label(value=f"{name}: None")
    clear_btn = PushButton(text="Clear")
    rotate_btn = PushButton(text="Rotate 90°")
    flipv_btn = PushButton(text="Flip V")
    fliph_btn = PushButton(text="Flip H")
    new_pts_btn = PushButton(text="New Points Layer")

    # Container for all widgets
    all_widgets = [Label(value=f"<h3>{name}</h3>")]

    # Add specific loader based on panel name
    if name == "Image 1":
        image_file_edit = FileEdit(label="", mode="r", filter="*.mrc *.tif *.tiff *.png *.jpg")
        load_image_button = PushButton(text="Load Image File")

        def _on_load_image_click():
            image_path = image_file_edit.value
            if not image_path or not Path(image_path).exists():
                print("Error: Please select an image file to load.")
                return
            try:
                if Path(image_path).suffix.lower() == ".mrc":
                    with mrcfile.open(str(image_path), permissive=True) as mrc:
                        data = np.copy(mrc.data)
                    layer = viewer.add_image(data, name=Path(image_path).stem, colormap="gray")
                else:
                    img = io.imread(str(image_path))
                    layer = viewer.add_image(img, name=Path(image_path).stem)
                
                # Manually refresh choices before setting the value
                refresh_choices()

                # Auto-assign the loaded image
                combo.value = layer.name
                assigned_images[name] = layer
                label.value = f"{name}: {layer.name}"

            except Exception as e:
                print(f"Error loading image {Path(image_path).name}: {e}")

        load_image_button.clicked.connect(_on_load_image_click)
        all_widgets.extend([image_file_edit, load_image_button])

    elif name == "Image 2":
        nav_file_edit = FileEdit(label="", mode="r", filter="*.nav")
        load_nav_button = PushButton(text="Load NAV Montage")

        def _on_load_nav_click():
            nav_path = nav_file_edit.value
            if not nav_path or not Path(nav_path).exists():
                print("Error: Please select a .nav file to load.")
                return

            maps, _ = parse_nav(nav_path)
            if not maps:
                print(f"Info: No maps found in {Path(nav_path).name}.")
                return

            for map_id, info in maps.items():
                if not info or "file" not in info:
                    print(f"Warning: Skipping map {map_id} due to missing file info.")
                    continue
                mrc_path = Path(nav_path).parent / info["file"]
                if not mrc_path.exists():
                    print(f"Warning: MRC file for map {map_id} not found at: {mrc_path}")
                    continue
                try:
                    montage = reconstruct_from_nav(mrc_path, info["coords"])
                    viewer.add_image(montage, name=f"Montage Map {map_id}", colormap="gray")
                except Exception as e:
                    print(f"Error: Failed to reconstruct montage for map {map_id}: {e}")

        load_nav_button.clicked.connect(_on_load_nav_click)
        all_widgets.extend([nav_file_edit, load_nav_button])


    # Add common widgets
    all_widgets.extend([combo, label, clear_btn,
                        rotate_btn, flipv_btn, fliph_btn,
                        new_pts_btn])

    def refresh_choices(event=None):
        current_choice = combo.value
        choices = [layer.name for layer in viewer.layers if isinstance(layer, Image)]
        combo.choices = choices
        if current_choice in choices:
            combo.value = current_choice

    viewer.layers.events.inserted.connect(refresh_choices)
    viewer.layers.events.removed.connect(refresh_choices)
    refresh_choices() # Initial population

    def on_select(event=None):
        if combo.value:
            assigned_images[name] = viewer.layers[combo.value]
            label.value = f"{name}: {combo.value}"

    def on_clear(event=None):
        assigned_images[name] = None
        assigned_points[name] = None
        label.value = f"{name}: None"
        combo.value = None

    def transform_image(fn):
        layer = assigned_images.get(name)
        if layer is None:
            return
        data = layer.data
        layer.data = fn(data)

        # Apply transformation also to linked points
        points_layer = assigned_points.get(name)
        if points_layer is not None:
            pts = points_layer.data.copy()
            if fn == np.flipud:
                pts[:, 1] = data.shape[0] - pts[:, 1]
            elif fn == np.fliplr:
                pts[:, 0] = data.shape[1] - pts[:, 0]
            elif fn.__name__ == "<lambda>":
                W = data.shape[1]
                new_pts = np.zeros_like(pts)
                new_pts[:, 0] = pts[:, 1]
                new_pts[:, 1] = W - pts[:, 0]
                pts = new_pts
            points_layer.data = pts

    def on_new_points(event=None):
        layer = viewer.add_points(
            np.empty((0, 2)),
            name=f"{name} Points",
            size=10,
            face_color="red" if name == "Image 1" else "blue"
        )
        assigned_points[name] = layer

    rotate_btn.clicked.connect(lambda e: transform_image(lambda d: np.rot90(d, k=1)))
    flipv_btn.clicked.connect(lambda e: transform_image(np.flipud))
    fliph_btn.clicked.connect(lambda e: transform_image(np.fliplr))
    new_pts_btn.clicked.connect(on_new_points)

    combo.changed.connect(on_select)
    clear_btn.clicked.connect(on_clear)

    return Container(widgets=all_widgets)


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
    return Container(widgets=[Label(value="<h3>Load Points</h3>"), file_edit, combo, button])
