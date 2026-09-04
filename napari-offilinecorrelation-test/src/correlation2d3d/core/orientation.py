import numpy as np
from correlation2d3d.core.geometry import   Points2D
from correlation2d3d.core.transform import apply_affine_matrix


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