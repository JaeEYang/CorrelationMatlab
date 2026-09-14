import numpy as np
from correlation2d3d.core.geometry import   Points2D
from correlation2d3d.core.transform import apply_affine_matrix
from skimage.transform import AffineTransform, warp


# gotta make this a class with diffren attributes and for each 

"""
horizontal_flip_matrix(W)
            │
            │
            ├───────────────┐
            ▼               ▼
    existing points     orientation history
         F @ p           F @ old_O
    
"""
def horizontal_flip_matrix( width: int) -> np.ndarray:
    return np.array([
        [-1.0, 0.0, width - 1.0], # basic math x' = -x + (w-1)
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float64)


    
# Vertical flip siimlary  as horizontal 
def vertical_flip_matrix(height: int) -> np.ndarray:
    return np.array([
        [1.0,  0.0, 0.0],
        [0.0, -1.0, height - 1.0],  # basic math y' = -y + (H-1)
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)
    

# shift the center to the origin, rotate, shift back
def rotation_matrix(height: int,width: int,angle_degrees: float) -> np.ndarray:

    # We are ALREADY on the rotation-safe canvas
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    # the slider gives us degrees but sin and cos need radians
    theta = np.deg2rad(
        angle_degrees
    )

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)


    # move the center to (0,0) so we rotate around the middle instead of a corner
    translate_to_origin = np.array([
        [1.0, 0.0, -center_x],
        [0.0, 1.0, -center_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

 
    # y goes down in image coordinates
    # with these signs a positive angle moves a point on the right upwards, so
    # counterclockwise
    # a negative angle goes clockwise in this coordinate system
    rotate = np.array([
        [ cos_theta, sin_theta, 0.0],
        [-sin_theta, cos_theta, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # put the center back where it was after rotating
    translate_back = np.array([
        [1.0, 0.0, center_x],
        [0.0, 1.0, center_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # read from the right, move to the origin -> rotate -> move back
    matrix = (
        translate_back
        @ rotate
        @ translate_to_origin
    )

    return matrix
    
# build complete orientation transform for an image without changing a single pixel

# This function is part of new refator so that we don't have to do the padding
#Here are the original pixels. when drawn on napari, place them according to this matrix

def orientation_matrix_from_settings(
    height: int,
    width: int,
    angle_degrees: float = 0.0,
    *,
    horizontal_flipped: bool = False,
    vertical_flipped: bool = False,
) -> np.ndarray:

    # error handling
    if height <= 0 or width <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    if not np.isfinite(angle_degrees):
        raise ValueError(
            "angle must be finite"
        )
    
    # build from the current settings each time, not from the old matrix
    # so changing 30 to 31 builds 31 directly, it doesn't add another rotation
    rotation = rotation_matrix(
        height,
        width,
        angle_degrees,
    )

    operation = rotation

    # put H on the left so the flip happens after rotation, H @ R
    # this flips in the plugin's coordinate frame
    # the controller can still apply a separate manual napari transform after this
    if horizontal_flipped:
        operation = (
            horizontal_flip_matrix(width)
            @ operation
        )

    # if vertical flip is on, put V on the left too, giving us V @ H @ R
    # just skip whichever flips are off
    if vertical_flipped:
        operation = (
            vertical_flip_matrix(height)
            @ operation
        )

    # return the matrix with full tranformation
    return operation
    



