from dataclasses import dataclass 
import numpy as np

#one place to establish invariants 
# shape of points is (N,2) 
#dtype of points is float64
#internal convention is (x,y)
#stored array cannot be mutated, so frozen dataclass is used
# could be used later as e.g fit_affine(source: Points2D, destination: Points2D) we know both the coordinates are the valid coordinate collections
# no min requirement here how many points are needed depends on the algorithm later used. affine >= 3
@dataclass(frozen=True,eq=False)  #   protects attribute assignment
class Points2D:
    xy:np.ndarray

    def __post_init__(self):
        array = np.array(
        self.xy, 
        dtype= np.float64,
        copy = True, # create a copy to avoid modifying the original array
        order = 'C' # store coordinates in C-contiguous memory
    )
        # make sure the dimension matches the expected shape (N, 2)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Points2D must be a 2D array with shape (N, 2), got shape {array.shape}")
        if not np.isfinite(array).all(): # finite values no NaN or inf. Could mess up the tranformation calculations
            raise ValueError("Points2D array must contain only finite values.")

        array.setflags(write=False)  # Make the Numpy buffer read only to prevent accidental mutation
        object.__setattr__(self, 'xy', array) # bypass frozen dataclass restriction to set the attribute
        
    def __len__(self):
        return len(self.xy)
    
    # Return a copy of the points in row-column order (y,x)
    def to_rc(self) -> np.ndarray:
        """Return a copy of the points in row-column order (y,x)"""
        return self.xy[:, ::-1].copy()  # reverse the columns to get (y,x) and return a copy
    
    # Return a copy of the points in x-y order (x,y)
    @classmethod
    def from_rc(cls, rc) -> "Points2D": 
        array = np.asarray(rc)

        if array.ndim != 2 or array.shape[1] != 2: # check if the input array has the correct shape (N, 2)
            raise ValueError(
                f"row/column coordinates must have shape (N, 2), "
                f"got shape {array.shape}"
            )

        return cls(array[:, ::-1])
    
    # mask should be 1-d, Boolean, same length as Points2D e.g [True, False, True, ...]
    def subset(self, mask) -> "Points2D":
        mask_array = np.asarray(mask)
        
        if mask_array.ndim != 1:
            raise ValueError("subset mask must be one-dimensional")

        if mask_array.dtype != np.bool_:
            raise TypeError("subset mask must contain Boolean values")

        if len(mask_array) != len(self):
            raise ValueError(
                "subset mask length must match the number of points"
            )
        return Points2D(self.xy[mask_array]) # return a new Points2D instance with the subset of points where mask is True
        
    

