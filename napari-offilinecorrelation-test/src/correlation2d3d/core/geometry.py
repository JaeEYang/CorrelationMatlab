from dataclasses import dataclass 
import numpy as np

#one place to establish invariants 
# shape of points is (N,2) 
#dtype of points is float64
#internal convention is (x,y)
#stored array cannot be mutated, so frozen dataclass is used
# could be used later as e.g fit_affine(source: Points2D, destination: Points2D) we know both the coordinates are the valid coordinate collections
# no min requirement here how many points are needed depends on the algorithm later used. affine >= 3
@dataclass(frozen=True,eq=False)
class Points2D:
    xy:np.ndarray

    def __post_init__(self):
        array = np.array(
        self.xy, 
        dtype= np.float64,
        copy = True,
        order = 'C'
    )

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Points2D must be a 2D array with shape (N, 2), got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Points2D array must contain only finite values.")

        array.setflags(write=False)  # Make the array immutable
        object.__setattr__(self, 'xy', array)
        
    def __len__(self):
        return len(self.xy)

