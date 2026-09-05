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

def flip_horizontal(image:np.ndarray, points:Points2D|None = None) -> tuple[np.ndarray, Points2D|None]:
    
    if image.ndim < 2:
        raise ValueError(
            "image must have atleast two dimensions"
        )
        
    width = image.shape[1] # this tells us how many colums we have
    #flip the image
    flipped_image = np.flip(
        image,
        axis=1 # flips horzontally reverses the columns
    ).copy()
    
    flipped_points = None
    
    if points is not None:
        flipped_points = apply_affine_matrix(
            horizontal_flip_matrix(width),
            points,
        )

    return (
        flipped_image,
        flipped_points,
    )
    
    
# Vertical flip siimlary  as horizontal 
def vertical_flip_matrix(height: int) -> np.ndarray:
    return np.array([
        [1.0,  0.0, 0.0],
        [0.0, -1.0, height - 1.0],  # basic math y' = -y + (H-1)
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)
    
def flip_vertical(image: np.ndarray, points: Points2D | None = None) -> tuple[np.ndarray, Points2D | None]:

        if image.ndim < 2:
            raise ValueError(
                "image must have at least two dimensions"
            )

        height = image.shape[0]

        flipped_image = np.flip(
            image,
            axis=0,
        ).copy()

        flipped_points = None
        # if points are loaded aswell at that point apply the same tranformation to them aswell.
        if points is not None: 
            flipped_points = apply_affine_matrix(
                vertical_flip_matrix(height),
                points,
            )

        return (
            flipped_image,
            flipped_points,
        )
    
def apply_orientation_to_points( points: Points2D, matrix: np.ndarray) -> Points2D:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "orientation matrix must have shape (3, 3)"
        )

    homogeneous = np.column_stack([
        points.xy,
        np.ones(len(points)),
    ])

    #orientation_matrix
    #original modality -> current modality
    # using the matrix calcuate the tranfomation.
    transformed = (
        matrix @ homogeneous.T
    ).T

    return Points2D(
        transformed[:, :2]
    )
    
def rotation_canvas_shape( height: int, width: int) -> tuple[int, int]:
    # error handling
    if height <= 0 or width <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    # get the diagonal size of the image
    diagonal = int(
        np.ceil(
            np.sqrt(
                height ** 2 + width ** 2
            )
        )
    )

    output_height = diagonal
    output_width = diagonal

    # if one is even and other is odd and vice versa
    # then match the output height parity to the input height.
    # we can also do the big wise operation here a bit too much though.
    if output_height % 2 != height % 2:
        output_height += 1

    # Match the output width parity to the input width.
    if output_width % 2 != width % 2:
        output_width += 1

    return (
        output_height,
        output_width,
    )

def prepare_rotation_canvas(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    if image.ndim < 2:
        raise ValueError(
            "image must have at least two dimensions"
        )

    height, width = image.shape[:2]

    output_height, output_width = rotation_canvas_shape(
        height,
        width,
    )

    # Because we matched odd/even parity,
    # these offsets are guaranteed to be integers.
    offset_y = (output_height - height) // 2
    offset_x = (output_width - width) // 2

    canvas_shape = (
        output_height,
        output_width,
        *image.shape[2:],
    )

    canvas = np.zeros(
        canvas_shape,
        dtype=image.dtype,
    )

    canvas[
        offset_y:offset_y + height,
        offset_x:offset_x + width,
    ] = image

    padding_matrix = np.array([
        [1.0, 0.0, offset_x],
        [0.0, 1.0, offset_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return (
        canvas,
        padding_matrix,
    )
    
def rotation_matrix(
    height: int,
    width: int,
    angle_degrees: float,
) -> tuple[np.ndarray, tuple[int, int]]:

    # We are ALREADY on the rotation-safe canvas.
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    theta = np.deg2rad(
        angle_degrees
    )

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    translate_to_origin = np.array([
        [1.0, 0.0, -center_x],
        [0.0, 1.0, -center_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    rotate = np.array([
        [ cos_theta, sin_theta, 0.0],
        [-sin_theta, cos_theta, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    translate_back = np.array([
        [1.0, 0.0, center_x],
        [0.0, 1.0, center_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    matrix = (
        translate_back
        @ rotate
        @ translate_to_origin
    )

    return (
        matrix,
        (height, width),
    )
    
def rotate_image(image: np.ndarray, angle_degrees: float,*, order: int = 1) -> tuple[np.ndarray, np.ndarray]:

    if image.ndim < 2:
        raise ValueError(
            "image must have at least two dimensions"
        )

    height, width = image.shape[:2]

    # get the rotation matrix
    matrix, output_shape = rotation_matrix(
        height,
        width,
        angle_degrees,
    )
    # apply the transformation
    transform = AffineTransform(
        matrix=matrix
    )

    rotated_image = warp(
        image,
        inverse_map=transform.inverse,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )

    return (
        rotated_image,
        matrix,
    )