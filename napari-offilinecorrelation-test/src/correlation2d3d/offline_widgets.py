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

def _resolve_map_path(nav_path: str, mapfile_field: str) -> Optional[Path]:
    """
    Resolve a NAV MapFile entry to a file on this machine.

    A NAV file records the path from the acquisition machine (typically
    something like X:\\RawData\\...), which usually does not exist wherever the
    NAV is later read. Resolution is deliberately limited to two deterministic
    steps so the outcome is predictable:

        1. the stored path, exactly as written
        2. the same filename sitting next to the NAV file

    Returning None is a normal outcome, not an error: the caller is expected to
    ask the user to locate the file.
    """
    stored = Path(_normalize_windows_path(mapfile_field))

    if stored.is_file():
        print(f"Map file found at its stored path: {stored}")
        return stored

    beside_nav = Path(nav_path).parent / stored.name
    if beside_nav.is_file():
        print(f"Map file found next to the NAV file: {beside_nav}")
        return beside_nav

    print(f"Could not locate {stored.name} automatically; asking the user.")
    return None


def _prompt_for_map_file(expected_name: str, start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Ask the user to locate a map file we could not resolve.

    Returns None if the user cancels, which callers must treat as "do nothing"
    rather than as an error.
    """
    chosen, _ = QFileDialog.getOpenFileName(
        None,
        f"Locate {expected_name}",
        str(start_dir) if start_dir else "",
        "Map images (*.st *.mrc *.mrcs *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
    )
    if not chosen:
        return None
    return Path(chosen)


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
    nav_edit = FileEdit(label="Template NAV", mode="r", filter="*.nav")

    nav_maps = []
    
    navdata = None  ## newly added

    # queued export state

    queued_points: List[Tuple[int, nb.PointItem]] = []
    queued_source_nav: Optional[str] = None
    queued_count_label = Label(value="Queued points: 0")

    def get_map_choices(widget=None):
        return [
            f"Map {m.Label} (ID={m.MapID}, Regis={m.Regis}, File={Path(m.MapFile).name if m.MapFile else 'None'})"
            for m in nav_maps
        ]
    
    def get_selected_map(combo: ComboBox):
        if combo.value is None:
            return None
        try:
            idx = get_map_choices(combo).index(combo.value)
            return nav_maps[idx]
        except ValueError:
            return None
    # ---Newly added: existing point groups for selected map ---

    def get_group_choices(widget=None):
        if navdata is None:
            return []
        
        map_item = get_selected_map(combo_1)
        if map_item is None:
            return []
        
        pts_for_map = [p for p in navdata.Points if p.DrawnID == map_item.MapID]
        if not pts_for_map:
            return []
        
        groups = {}
        for p in pts_for_map:
            groups.setdefault(p.GroupID, []).append(p)
        labels = []
        for gid in sorted(groups.keys()):
            n = len(groups[gid])
            labels.append(f"GroupID = {gid} ({n} pts)")
        return labels
    
    #---Newly added--ending

    combo_1 = ComboBox(label="Select Map from NAV", choices=get_map_choices)
    combo_2 = ComboBox(label="Assign to Map", choices=get_map_choices)

    #--Newly added--starting
    existing_group_combo = ComboBox(
        label="Existing Point Groups",
        choices=get_group_choices,
    )
    #---Newly added--ending

    btn_show_map = PushButton(text="Show Map")
        # --- Optional registration files (FLM ↔ TEM)
    flm_reg_pts_edit = FileEdit(
        label="FLM Registration Points (CSV)",
        mode="r",
        filter="*.csv",
        tooltip="Optional: select FLM registration points file"
    )
    tem_reg_pts_edit = FileEdit(
        label="TEM Registration Points (CSV)",
        mode="r",
        filter="*.csv",
        tooltip="Optional: select TEM registration points file"
    )
    def _on_flm_csv_selected(event=None):
        btn_view_flm.enabled = bool(flm_reg_pts_edit.value)
    def _on_tem_csv_selected(event=None):
        btn_view_tem.enabled = bool(tem_reg_pts_edit.value)

    flm_reg_pts_edit.changed.connect(_on_flm_csv_selected)
    tem_reg_pts_edit.changed.connect(_on_tem_csv_selected)
    csv_edit = FileEdit(label="Points CSV", mode="r", filter="*.csv")
    btn_view = PushButton(text="View Points")
    btn_view_flm = PushButton(text="Preview FLM regPoints")
    btn_view_tem = PushButton(text="Preview TEM regPoints")
    btn_warp = PushButton(label="Transform FLM → TEM")
    #btn_add = PushButton(text="Add Points to NAV")
    btn_add = PushButton(text="Add Points to NAV (queue)")
    btn_write = PushButton(text="Write NAV")

    

    ## forcing the order
    combo_1.enabled = True
    btn_show_map.enabled = True
    csv_edit.enabled = True
    btn_view.enabled = True
    btn_view_flm.enabled = True
    btn_view_tem.enabled = True
    combo_2.enabled = True
    btn_warp.enabled = True
    btn_add.enabled = True

    btn_write.enabled = True
    existing_group_combo.enabled = True

    # --- Step 1: Load NAV file 
    def _on_nav_change(event=None):
        nonlocal navdata # newly added
        nav_path = nav_edit.value
        if not nav_path or not Path(nav_path).exists():
            print("Please select a valid NAV file.")
            return
        navdata = nb.parseNavFile(str(nav_path))
        if navdata and navdata.Maps:
            nav_maps.clear()
            nav_maps.extend(navdata.Maps)
            print("DEGUG CHOICES:", get_map_choices)
            combo_1.reset_choices()
            combo_2.reset_choices()
            combo_1.enabled = True
            combo_2.enabled = True
            existing_group_combo.enabled = True # newly added
            btn_show_map.enabled = True
            print(f" Loaded {len(navdata.Maps)} map(s) from {nav_path}")

    nav_edit.changed.connect(_on_nav_change)
    
    # --- Step 2-3: Show Map (combo_1)

    def _on_show_map(event=None):
        nav_path = nav_edit.value   
        if not nav_path or not Path(nav_path).exists():
            print("⚠ Template NAV file not found")
            return

        map_item = get_selected_map(combo_1)
        if map_item is None:
            print("no map selected")
            return

        # Use the path stored in the NAV if it exists; otherwise ask the user to
        # locate the file. NAV files carry acquisition-machine paths, so on any
        # other machine the prompt is the normal route rather than an error case.
        map_path = _resolve_map_path(nav_path, map_item.MapFile)

        if map_path is None:
            expected_name = Path(_normalize_windows_path(map_item.MapFile)).name
            map_path = _prompt_for_map_file(expected_name, start_dir=Path(nav_path).parent)
            if map_path is None:
                print(f"Cancelled: {expected_name} was not located.")
                return
        if map_path.is_dir():
            print(f"⚠ Resolved map_path is a directory, not a file: {map_path}")
            return
        
        print(f"ℹ Using map file: {map_path}")

        try:
            mdoc_path = Path(str(map_path) + ".mdoc")
            is_montage = mdoc_path.exists()
            if is_montage:
                print(f"ℹ Found MDOC file: {mdoc_path}")
                try:
                    arr = reconstruct_mdoc_montage(map_path, mdoc_path) #Use the function to display montage
                    print(f"✅ Reconstructed montage from MDOC | shape={arr.shape}")
                except Exception as e:
                    print(f"⚠ Failed MDOC reconstruction, falling back to raw stack: {e}")
                    arr = _read_map_array(map_path)
                    is_montage = False

            else:
                # 2) No MDOC → fallback to raw reading
                arr = _read_map_array(map_path)
                print(f"ℹ Loaded raw map image | shape={arr.shape}")

            # ---------------------------------------------
            # Display reconstructed or raw image
            # ---------------------------------------------
            layer_name = f"Map {map_item.Label}"

            if layer_name in viewer.layers:
                viewer.layers.remove(layer_name)

            if is_montage:

                canvas, offset = prepare_canvas(arr)
                offsets[layer_name] = offset
                layer = viewer.add_image(canvas, name=layer_name, colormap="gray")
                layer.translate = np.array([offset[0], offset[1]])
                print(f"⭐ Montage displayed. Offset={offset}")
            else:
                offsets[layer_name] = (0, 0)
                layer = viewer.add_image(arr, name=layer_name, colormap="gray")
                print("⭐ Non-montage image displayed. No offset applied.")

            viewer.camera.center = (
                layer.extent.world[0]
                + (layer.extent.world[1] - layer.extent.world[0]) / 2
            )
            print(f"✅ Loaded map {map_item.Label} from {map_path}")

            ## temporiraly added for testing
            sx, sy = map_item.StageXYZ[:2]
            xp, yp = _stage_to_pixel(map_item, sx, sy)
            sx2, sy2 = sn.pixel_to_stage(map_item, xp, yp)
            print(f"  original stage : ({sx:.6f}, {sy:.6f})")
            print(f"  pixel coords   : ({xp:.6f}, {yp:.6f})")
            print(f"  recovered stage: ({sx2:.6f}, {sy2:.6f})")
            print(f"  Δstage         : ({sx2 - sx:.3e}, {sy2 - sy:.3e})")

        except Exception as e:
            print(f"⚠ Failed to read map image from {map_path}: {e}")
            return
        print(">>> REACHED END OF SHOW MAP — ENABLE UI")

        # testing on map coordinates
        corners = _test_map_corners(map_item)
        viewer.add_shapes(
            [corners],
            shape_type="polygon",
            edge_color="lime",
            face_color=[0, 0, 0, 0],
            name="Map corner test"
        )

    btn_show_map.clicked.connect(_on_show_map)

    ##newly added, helper
    def _stage_to_pixel(map_item, stage_x, stage_y):
        M = np.array(map_item.MapScaleMat, dtype=float) 
        raw = np.array(map_item.StageXYZ[:2], dtype=float) #raw = np.array(map_item.RawStageXY, dtype=float)
        WH = np.array(map_item.MapWidthHeight, dtype=float)
        cen = np.array([WH[0] * 0.5, WH[1] * 0.5])
        vec = np.array([stage_x, stage_y], dtype=float) - raw
        pix = M @ vec + cen  #pix = vec @ M + cen
        # added one line to test
        pix[1] = WH[1] - pix[1]

        # debugging printout
        print("MapScaleMat:\n", M)
        print("vec:", vec)
        print("pix:", pix)

        return float(pix[0]), float(pix[1])
    
    # newly added temporiraly for testing
    def _test_map_corners(map_item):
        W, H = map_item.MapWidthHeight
        corners_stage = list(zip(map_item.PtsX[:4], map_item.PtsY[:4]))
        corners_pix = []
        for i, (sx, sy) in enumerate(corners_stage):
            xpix, ypix = _stage_to_pixel(map_item, sx, sy)
            corners_pix.append([ypix, xpix])
            print(f"Corner {i}: stage=({sx:.3f}, {sy:.3f}) → pixel=({xpix:.1f}, {ypix:.1f})")
        
        return np.array(corners_pix, dtype=float)

    # --- Step 4: Load CSV and preview points
    def _on_view(event=None):

        map_item = get_selected_map(combo_1)
        if map_item is None:
            print("No map selected.")
            return
        
        ## newly added, Group points selected
        group_label = existing_group_combo.value
        if group_label:
            import re
            m = re.search(r"GroupID\s*=\s*(-?\d+)", group_label)
            if not m:
                print(f"⚠ Could not parse GroupID from '{group_label}'")
                return
            gid = int(m.group(1))
            print(f"Selected GroupID = {gid}")

    

            pts_for_group = [
                p for p in navdata.Points
                if p.DrawnID == map_item.MapID and p.GroupID == gid
            ]
            if not pts_for_group:
                print(f"No points found for Group {gid} on Map {map_item.Label}")
                return
            px = []
            for p in pts_for_group:
                Xpix, Ypix = _stage_to_pixel(map_item, p.StageXYZ[0], p.StageXYZ[1])
                px.append([Ypix, Xpix])
            pts_arr = np.array(px, dtype=float)

            
            layer_name = f"GroupID_{gid}_Points"
            if layer_name in viewer.layers:
                viewer.layers[layer_name].data = pts_arr
            else:
                viewer.add_points(
                    pts_arr,
                    name=layer_name,
                    size=14,
                    face_color="magenta",
                )
            print(f"Displayed {len(pts_arr)} existing NAV points for GroupID {gid}")
            return
        
        ## No Group selected
        csv_path = csv_edit.value
        if not csv_path or not Path(csv_path).exists():
            print(" CSV file not found")
            return

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            # remove blank lines and BOM (\ufeff) safely
            lines = [line for line in f if line.strip()]

        coords = np.loadtxt(lines, delimiter=",")

        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        if coords.shape[1] < 2:
            print(" CSV must have at least 2 columns (X,Y)")
            return
        
        # Convert [X, Y] → [Y, X] for napari display
        pts = coords[:, [1, 0]].copy()  
      
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

    btn_view.clicked.connect(_on_view)

    def _load_and_display_csv(csv_path, layer_label):
        if not csv_path or not Path(csv_path).exists():
            print(f"⚠ CSV file not found for {layer_label}")
            return
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            lines = [line for line in f if line.strip()]
        
        coords = np.loadtxt(lines, delimiter=",")
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.shape[1] < 2:
             print(f"⚠ {layer_label}: CSV must have at least 2 columns (X,Y)")
             return
        
        pts = coords[:, [1, 0]]

        if layer_label in viewer.layers:
            viewer.layers[layer_label].data = pts
        else:
            viewer.add_points(
                pts,
                name=layer_label,
                size=12,
                face_color="red" if "FLM" in layer_label else "cyan",
            )
        
        print(f"Displayed {pts.shape[0]} points for {layer_label} from {csv_path}")
    
    def _on_view_flm(event=None):
        csv_path = flm_reg_pts_edit.value
        print("FLM CSV PATH =", csv_path)
        _load_and_display_csv(csv_path, "FLM Registration Points")
    
    btn_view_flm.clicked.connect(_on_view_flm)
    
    def _on_view_tem(event=None):
        csv_path = tem_reg_pts_edit.value
        _load_and_display_csv(csv_path, "TEM Registration Points")

    btn_view_tem.clicked.connect(_on_view_tem)

    def get_points_layer_choices(widget=None):
        return [layer.name for layer in viewer.layers if layer.__class__.__name__ == "Points"]
    
    points_layer_combo = ComboBox(
        label="Select Point Layer",
        choices=get_points_layer_choices,
    )
    flm_reg_points_combo = ComboBox(
        label="FLM reg point layer (optional)",
        choices=get_points_layer_choices,
    )
    tem_reg_points_combo = ComboBox(
        label="TEM reg point layer (optional)",
        choices=get_points_layer_choices,
    )

    def get_image_layer_choices(widget=None):
        return [layer.name for layer in viewer.layers if layer.__class__.__name__ == "Image"]
    
    flm_image_combo = ComboBox(
        label="Select FLM Image Layer",
        choices=get_image_layer_choices,
    )

    tem_image_combo = ComboBox(
        label="Select TEM Image Layer",
        choices=get_image_layer_choices,
    )


    # --- Step 5: Add points into NAV (writing out proper nav)
    def computeMsFLM(flm_pts, tem_pts):
        import numpy as np

        flm_pts = np.asarray(flm_pts)
        tem_pts = np.asarray(tem_pts)

        if flm_pts.shape != tem_pts.shape:
            raise ValueError("FLM and TEM point arrays must have the same shape.")

        if flm_pts.shape[0] < 3:
            raise ValueError("At least 3 point pairs are required.")

        N = flm_pts.shape[0]

        # Build 3×N homogeneous coordinate matrices (COLUMN VECTORS)
        P = np.vstack([
            flm_pts[:, 0],          # x_f
            flm_pts[:, 1],          # y_f
            np.ones(N)
        ])                          # shape (3, N)

        Q = np.vstack([
            tem_pts[:, 0],          # x_t
            tem_pts[:, 1],          # y_t
            np.ones(N)
        ])                          # shape (3, N)

        rank = np.linalg.matrix_rank(P)
        if rank < 3:
            print(f"⚠ Registration points are rank-deficient (rank={rank}); using pseudoinverse fit.")
        # Compute the column-based least-squares transform
        #M_col = Q @ P.T @ np.linalg.inv(P @ P.T)   # shape (3, 3), strict affine shape
        M_col = Q @ P.T @ np.linalg.pinv(P @ P.T)   # proper for interactive layers

    # Convert to row-vector
        MsFLM_row = M_col.T
        return MsFLM_row
    
    def _resolve_registration_points():
        flm_layer_name = flm_reg_points_combo.value
        tem_layer_name = tem_reg_points_combo.value

        if flm_layer_name and tem_layer_name:
            if flm_layer_name not in viewer.layers or tem_layer_name not in viewer.layers:
                raise ValueError("Selected registration point layers not found")
            flm_layer = viewer.layers[flm_layer_name]
            tem_layer = viewer.layers[tem_layer_name]
            flm_pts_yx = flm_layer.data[:, :2].copy()
            tem_pts_yx = tem_layer.data[:, :2].copy()
            if flm_pts_yx.shape != tem_pts_yx.shape or flm_pts_yx.shape[0] < 3:
                raise ValueError("Point layers must have same number of points (≥3)")
            print("Using registration points from napari point layers")
            source = "layers"

        else:
            flm_path = flm_reg_pts_edit.value
            tem_path = tem_reg_pts_edit.value

            if not (flm_path and tem_path):
                raise ValueError("Select BOTH point layers or provide BOTH CSV files for registration")
            flm_raw = np.loadtxt(flm_path, delimiter=",")
            tem_raw = np.loadtxt(tem_path, delimiter=",")
            if flm_raw.shape != tem_raw.shape or flm_raw.shape[0] < 3:
                raise ValueError("CSV registration files must match and have ≥3 points")
            
            flm_pts_yx = flm_raw[:, :2][:, ::-1]
            tem_pts_yx = tem_raw[:, :2][:, ::-1]
            print("Using registration points from CSV files")
            source = "csv"
        
        flm_img_name = flm_image_combo.value
        tem_img_name = tem_image_combo.value
        if not flm_img_name or not tem_img_name:
            raise ValueError("Select FLM and TEM image layers first")
        
        y0f, x0f = offsets.get(flm_img_name, (0, 0))
        y0t, x0t = offsets.get(tem_img_name, (0, 0))

        flm_pts_yx[:, 0] -= y0f
        flm_pts_yx[:, 1] -= x0f
        tem_pts_yx[:, 0] -= y0t
        tem_pts_yx[:, 1] -= x0t

        flm_pts = flm_pts_yx[:, ::-1]
        tem_pts = tem_pts_yx[:, ::-1]

        return flm_pts, tem_pts, source

    def warp_image_row_affine(src, Ms_row, output_shape):
        import numpy as np
        from scipy.ndimage import affine_transform
        Ms_inv = np.linalg.inv(Ms_row)
        A = Ms_inv[:2, :2].T
        t = Ms_inv[2, :2]
        A_rc = A[[1, 0], :][:, [1, 0]]
        t_rc = t[[1, 0]]
        if src.ndim == 2:
            warped = affine_transform(
                src,
                matrix=A_rc,
                offset=t_rc,
                output_shape=output_shape,
                order=1,
                mode="constant",
                cval=0.0
            )
        else:
            warped = np.zeros(output_shape + (src.shape[2],),
                              dtype=src.dtype)
            for c in range(src.shape[2]):
                warped[..., c] = affine_transform(
                    src[..., c],
                    matrix=A_rc,
                    offset=t_rc,
                    output_shape=output_shape,
                    order=1,
                    mode="constant",
                    cval=0.0
                )
        return warped
    
    def _on_warp(event=None):

        try:
            flm_pts, tem_pts, _ = _resolve_registration_points()
        except Exception as e:
            print(f"⚠ {e}")
            return
  
        try:
            MsFLM = computeMsFLM(flm_pts, tem_pts)
            print("Computed MsFLM:\n", MsFLM)

            # debugging
            pred = np.c_[flm_pts, np.ones(len(flm_pts))] @ MsFLM
            viewer.add_points(
                pred[:, ::-1],  # back to (y,x)
                name="Predicted TEM points",
                face_color="lime",
                size=20
            )

            # debugging
            i = 0  # test the first registration pair
            xf, yf = flm_pts[i]
            xt, yt = tem_pts[i]
            pred = np.array([xf, yf, 1.0]) @ MsFLM
            print(f"FLM input      : ({xf:.3f}, {yf:.3f})")
            print(f"TEM expected   : ({xt:.3f}, {yt:.3f})")
            print(f"TEM predicted  : ({pred[0]:.3f}, {pred[1]:.3f})")
            print(f"Residual (px)  : ({pred[0]-xt:.3e}, {pred[1]-yt:.3e})")

        except Exception as e:
            print(f"⚠ Failed to compute MsFLM: {e}")
            return

        flm_layer_name = flm_image_combo.value
        tem_layer_name = tem_image_combo.value

        if flm_layer_name not in viewer.layers:
            print(f"⚠ FLM layer '{flm_layer_name}' is not loaded.")
            return
        flm_image = viewer.layers[flm_layer_name].data
        
        if tem_layer_name not in viewer.layers:
            print(f"⚠ TEM layer '{tem_layer_name}' not found.")
            return
        tem_image = viewer.layers[tem_layer_name].data

        H, W = tem_image.shape[:2]
        output_shape = (H, W)
        flm_warped = warp_image_row_affine(
            flm_image,
            MsFLM,
            output_shape
        )

        viewer.add_image(
            flm_warped,
            name=f"{flm_layer_name}_warped_to_TEM",
            colormap="green",
            opacity=0.4
        )

    def _on_add(event=None):
        nonlocal queued_points, queued_source_nav

        MsFLM = None
        apply_MsFLM = False
        csv_path = csv_edit.value
        nav_path = nav_edit.value

        if not nav_path or not Path(nav_path).exists():
            print(" Template NAV file not found")
            return
        
        # parse Nav file using NavBuilt
        navdata = nb.parseNavFile(str(nav_path))

        # assign associated map layer
        assign_map_item = get_selected_map(combo_2)
        if assign_map_item is None:
            print("no assign map selected")
            return

        # assign associated point layer
        
        selected_layer_name = points_layer_combo.value
        if not selected_layer_name or selected_layer_name not in viewer.layers:
            print("No point layer selected")
            return
        coords = viewer.layers[selected_layer_name].data

        # Load CSV coords
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        # If queue is empty or template changed, reset numbering from template
        if (queued_source_nav is None) or (Path(queued_source_nav) != Path(nav_path)) or (len(queued_points) == 0):
            queued_points = []
            queued_source_nav = str(nav_path)
            last_item_num = get_last_item_number(nav_path)
        else:
            last_item_num = max(item_num for item_num, _ in queued_points)

        # --- FIX: get last item number once from template
        #last_item_num = get_last_item_number(nav_path)
        import random
        new_group_id = random.randint(1_000_000_000, 9_999_999_999)
        
        #print(f"Last item number found in template NAV = {last_item_num}")


        new_points = []

        for offset, row in enumerate(coords, start=1):
            if len(row) == 2:
                y, x = row
                z = 0.0
            else:
                y, x, z = row[:3]

            item_num = last_item_num + offset  # stable numbering

            ## where original block starts
            pixel_vec = np.array([x, y, 1.0])
            if MsFLM is not None:
                transformed_pixel = pixel_vec @ MsFLM
                x, y = transformed_pixel[:2]
            
            # Then convert pixel to stage using MapScaleMat

            H = float(assign_map_item.MapWidthHeight[1])
            y = H - y

            stage_xy = sn.pixel_to_stage(assign_map_item, x, y)
            stage_x, stage_y = stage_xy[0], stage_xy[1]
            point_map_id = random.randint(1_000_000_000, 9_999_999_999)


            # Create a Nav point item for each point
            p = nb.PointItem()
            p.Label = str(item_num)
            p.StageXYZ = [float(stage_x), float(stage_y), float(z)]
            p.PtsX = float(stage_x)
            p.PtsY = float(stage_y)
            p.DrawnID = assign_map_item.MapID 
            p.Regis = assign_map_item.Regis
            p.GroupID = new_group_id
            p.MapID = point_map_id
            new_points.append((item_num, p))
        
        queued_points.extend(new_points)
        queued_count_label.value = f"Queued points: {len(queued_points)}"
        print(f"Queued {len(new_points)} point(s) onto MapID={assign_map_item.MapID}. Total queued={len(queued_points)}")

    def _on_write(event=None):
        nonlocal queued_points, queued_source_nav

        template = queued_source_nav or nav_edit.value

        if not template or not Path(template).exists():
            print("Template NAV file for queued points is missing.")
            return
        
        out_path, _ = QFileDialog.getSaveFileName(
            None, "Save Output NAV", "output.nav", "NAV Files (*.nav)"
        )
        if not out_path:
            print("⚠ Save cancelled")
            return
        
        # Write out new NAV
        with open(out_path, "w") as f:
            with open(template, "r") as fin:
                f.write(fin.read()) # preserve everything

            for item_num, p in queued_points:
                lines = p.getText()[:]
                if lines:
                    if lines[0].startswith("[Item"):
                        lines[0] = f"[Item = {item_num}]"
                    else:
                        lines.insert(0, f"[Item = {item_num}]")

                f.write("\n\n")
                f.write("\n".join(lines))

        print(f" NAV written: {out_path} (queued points={len(queued_points)})")

        combo_1.enabled = True
        combo_2.enabled = True
        existing_group_combo.enabled = True #newly added

    # Hook up buttons
    btn_warp.clicked.connect(_on_warp)
    btn_add.clicked.connect(_on_add)
    btn_write.clicked.connect(_on_write)
    
    # Return container with all buttons
    return Container(
        widgets=[nav_edit, 
                 combo_1,              #Select Map from Nav
                 btn_show_map,         #Show Map
                 existing_group_combo, #Existing Point Group
                 flm_reg_points_combo, #FLM reg point layer (optional)
                 tem_reg_points_combo, #TEM reg point layer (optional)
                 flm_reg_pts_edit,     #FLM Registration Points (CSV)
                 tem_reg_pts_edit,     #TEM Registration Points (CSV)
                 btn_view_flm,         #Preview FLM regPoints
                 btn_view_tem,         #Preview TEM regPoints
                 points_layer_combo,   #Select Point Layer
                 csv_edit,             #Point CSV
                 btn_view,             #View Points
                 flm_image_combo,      #Select FLM Image Layer
                 tem_image_combo,      #Select TEM Image Layer
                 combo_2,              #Assign to Map
                 btn_warp,             #Transform FLM to TEM
                 btn_add,
                 queued_count_label,
                 btn_write]              #Add to Nav
    )

# ---------------------------
# Montage testing
# ---------------------------

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
                    print(f"Mdoc failed: {e}")

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



