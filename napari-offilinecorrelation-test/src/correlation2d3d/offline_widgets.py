import numpy as np
from pathlib import Path
from magicgui.widgets import FileEdit, PushButton, ComboBox, Container, Label
import mrcfile
from skimage import io
from napari.layers import Image
from correlation2d3d import NavBuilt as nb
from correlation2d3d import SupportNav as sn
from qtpy.QtWidgets import QFileDialog
from napari.viewer import Viewer
import re
from pathlib import Path
from typing import List, Tuple, Optional
import os

# Global state: linked images and points
assigned_images = {"Image 1": None, "Image 2": None}
assigned_points = {"Image 1": None, "Image 2": None}

# ---------------------------
# NAV parsing + reconstruction
# ---------------------------

def get_last_item_number(nav_path: str) -> int:
    """
    Scan NAV file and return the highest [Item = N] number.
    """
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
    """Yield paths in root whose final name matches `name` case-insensitively."""
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
    """
    Resolves a MapFile field from a NAV to a local existing file.

    Strategy:
      1) Exact path after normalization
      2) Relative to NAV directory
      3) Apply prefix remaps (e.g., 'X:/RawData/...' -> '/Users/.../CorrelationMatlab/data')
      4) Case-insensitive filename search in candidate roots
      5) Stem + common extension search
    """
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
def make_image_panel(viewer: "napari.viewer.Viewer", name: str = "Image 1") -> Container:
    combo = ComboBox(label=f"Select {name}", choices=[])
    label = Label(value=f"{name}: None")
    clear_btn = PushButton(text="Clear")
    rotate_btn = PushButton(text="Rotate 90°")
    flipv_btn = PushButton(text="Flip V")
    fliph_btn = PushButton(text="Flip H")
    new_pts_btn = PushButton(text="New Points Layer")

    def refresh_choices(event=None):
        combo.choices = [layer.name for layer in viewer.layers if isinstance(layer, Image)]

    viewer.layers.events.inserted.connect(refresh_choices)
    viewer.layers.events.removed.connect(refresh_choices)

    def on_select(event=None):
        if combo.value:
            assigned_images[name] = viewer.layers[combo.value]
            label.value = f"{name}: {combo.value}"

    def on_clear(event=None):
        assigned_images[name] = None
        assigned_points[name] = None
        label.value = f"{name}: None"

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

    return Container(widgets=[combo, label, clear_btn,
                              rotate_btn, flipv_btn, fliph_btn,
                              new_pts_btn])


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
