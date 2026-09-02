from dataclasses import dataclass

import numpy as np

from correlation2d3d.core.geometry import Points2D

# store the affine transformation matrix as a 3x3 numpy array, where the last row is [0, 0, 1]
@dataclass(frozen=True, eq=False)
class Registration2D:
    matrix: np.ndarray
    forward_residuals: np.ndarray | None = None
    
    def __post_init__(self):
        matrix = np.array(
            self.matrix,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        
        # registration matrix must be 3x3, finite, and affine (last row [0, 0, 1])
        if matrix.shape != (3, 3):
            raise ValueError(
                "Registration2D matrix must have shape (3, 3)"
            )
        
        if not np.isfinite(matrix).all():
            raise ValueError(
                "Registration2D matrix must contain only finite values"
            )

        if not np.allclose(
            matrix[2],
            np.array([0.0, 0.0, 1.0]),
            atol=1e-12,
        ):
            raise ValueError(
                "Registration2D matrix must be affine with "
                "last row [0, 0, 1]"
            )

        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

        
    def apply(self, points: Points2D) -> Points2D:
        return _apply_matrix( 
            self.matrix,
            points,
        )
    @property # turns the method into a property so we can access it like an attribute e.g registration.rmse instead of registration.rmse()
    def rmse(self) -> float | None:
        if self.forward_residuals is None:
            return None

        return float(
            np.sqrt(
                np.mean(self.forward_residuals ** 2)
            )
        )
        
        
# Helper function to apply a 3x3 affine transformation matrix to a set of 2D points      
def _apply_matrix(matrix: np.ndarray, points: Points2D) -> Points2D:
    homogeneous = np.column_stack([
        points.xy,
        np.ones(len(points)),
    ]) 
    
    
    '''
    create homogeneous coordinates by appending a column of ones to the points 
    N × 3
    our points are in the form of 
    [x1 y1 1]
    [x2 y2 1]
    [x3 y3 1]
    convention I decided on q = M @ p , qhwew one point is column
    [x]
    [y]
    [1]
        so homogeneous.T changes Nx3 into 3xN, then we can multiply by the 3x3 matrix, then transpose back to Nx3
    
    Now each point is a column:

        point1   point2   point3

        x1       x2       x3
        y1       y2       y3
        1        1        1
        
    Then:

    self.matrix @ homogeneous.T

    has shapes:

    (3,3) @ (3,N)
            ↓
        (3,N)

    and transforms every point at once.

    Then the final:

    .T

    puts us back into our normal Points2D storage: (N,3)
    
    Finally:

    transformed[:, :2] means:

    take every row, but only columns 0 and 1

    so:

    [x', y', 1]

    becomes:

    [x', y']

    and we construct a new Points2D.
    
    '''
    
    transformed = (
        matrix @ homogeneous.T
    ).T

    return Points2D(transformed[:, :2])

# fit an affine transformation from source points to destination points
def fit_affine(
    source: Points2D,
    destination: Points2D,
) -> Registration2D:
    if len(source) != len(destination):
        raise ValueError(
            "source and destination must contain the same number of points"
        )
    # require at least 3 points to determine an affine transformation
    if len(source) < 3:
        raise ValueError(
            "affine registration requires at least 3 point correspondences"
        )
    # create a design matrix by appending a column of ones to the source points
    design = np.column_stack([
        source.xy,
        np.ones(len(source)),
    ])
    # use least squares to solve for the affine transformation coefficients
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        destination.xy,
        rcond=None,
    )
    # check if the rank of the design matrix is sufficient to determine a unique affine transformation (points must not be collinear)
    if rank < 3:
        raise ValueError(
            "source points do not contain enough independent geometry "
            "to determine an affine transform"
        )

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = coefficients.T
    
    # this part is for the residuals, we can calculate the predicted destination points using the fitted matrix and then compute the residuals
    predicted = _apply_matrix(
    matrix,
    source,
    )
    
    '''
    error shape:
    (N, 2)
    because every landmark has:
    [error_in_x, error_in_y]
    '''
    errors = predicted.xy - destination.xy # computer for each point the difference between the predicted and actual destination points
    forward_residuals = np.linalg.norm( # don't want two seperate error values per landmark, so norm (euclidean) 
        errors,
        axis=1, # this mean we want the norm across the columns, so for each row (landmark) we get a single value
    )

    return Registration2D(
        matrix=matrix,
        forward_residuals=forward_residuals,
    )