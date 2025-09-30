import numpy as np
from magicgui import magic_factory
from napari.types import ImageData, PointsData
from skimage import io
import numpy as np
from pathlib import Path

# --- function 1: Add random image ---
@magic_factory(call_button="Add random image")
def random_image_widget() -> ImageData:
    """Return a random 2D image for testing."""
    data = np.random.random((256, 256))
    return data

# --- function 2: Load FLM Image + Registration Points ---
@magic_factory
def load_flm_with_points(viewer: "napari.viewer.Viewer",
                         flm_path: Path,
                         points_path: Path = None) -> ImageData:
    # load FLM image
    img = io.imread(str(flm_path))
    if img.ndim == 2:
        viewer.add_image(img, name="FLM Image", colormap="grey")
    elif img.ndim == 3 and img.shape[2] in (3, 4): 
        viewer.add_image(img, name="FLM Image", rgb=True)
    else:
        raise ValueError(f"Unexpected image shape, exptected grey or RGB 2D")
    
    # load FLM registration points
    if points_path is not None and Path(points_path).exists():
        pts = np.loadtxt(str(points_path), delimiter=",")
        if pts.shape[0] == 3:
            pts = pts.T
        elif pts.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Points expected to be n x 3 or 3 x n format")
        
        if pts.shape[1] == 3:
            pts2d = pts[:, :2]
        else:
            pts2d = pts

        viewer.add_points(pts2d, name="Registration Points", size=10, face_color="red")
    
    return None
