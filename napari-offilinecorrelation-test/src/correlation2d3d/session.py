from dataclasses import dataclass, field
import numpy as np

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.transform import Registration2D


"""        CorrelationSession
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
   Points2D    Registration2D    images
 immutable       immutable       arrays

        session itself is mutable 
        
        
        
              CorrelationSession
                      |
             ┌────────┴────────┐
             │                 │
            FLM               TEM
             │                 │
          image             image
          points            points
       orientation       orientation
             │                 │
             └──────┬──────────┘
                    │
               registration
              
      
# One place to answer what is the CURRENT STATE OF EVERYTHING ? this becomes home for the changing states.
        

    
Mutable state for one imaging modality.

original_image:
    Copy of the source Image layer pixel data when the role is assigned.

image:
    Source pixel data used by the correlation workflow.
    Orientation and rough alignment do not modify this array; they are
    represented geometrically by the napari Image layer transform.

original_points:
    Landmark coordinates in source-image pixel coordinates.

points:
    Landmark coordinates in the current napari world coordinate system.

orientation_matrix:
    Plugin-owned rotation/flip transform in x/y coordinates.
    The complete current source transform, including native napari
    translation, rotation, and scale, is read from the assigned Image layer.
"""
@dataclass
class ModalityState:
 
    # None means we haven't assigned an image to this role yet
    # assignment copies the pixels into both arrays, orientation only changes
    # transforms
    original_image: np.ndarray | None = None 
    image: np.ndarray | None = None 
    # original here means source-image coordinates, not an untouched copy of the CSV
    # when landmarks are edited we work these out again using the inverse transform
    original_points: Points2D | None = None 
    points: Points2D | None = None
    
    #  this is not a regular attribute so we need to used field for customizing and wrap the configuration inside it 
    # intitailly the orientation matrix is doing nothing which is just identity 
    # numpy array are mutable objects by default 
    # default_factory bascially sayds : Call this function separately every time a new object is constructed. not point towards same. it expects function as its argument
    # lambda is a tiny function which returns the identity.
    orientation_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(
            3,
            dtype=np.float64,
        )
    )

   # Plugin-owned absolute rotation angle and flip settings.
    rotation_angle: float = 0.0
    horizontal_flipped: bool = False
    vertical_flipped: bool = False



"""Current state of one FLM-TEM correlation workflow."""
# including the images, points, and registration information. 
# this will me mutable because the state is expected to change as the user interacts with the application, e.g. loading images, selecting points, and computing registrations.
@dataclass
class CorrelationSession:
   #Every new session gets its own FLM ModalityState object and its own TEM ModalityState object.
    flm: ModalityState = field(
        default_factory=ModalityState
    )
    tem: ModalityState = field(
        default_factory=ModalityState
    )
    # store the fit here once it succeeds, invalidation puts it back to None
    # the widget and controller also clear the displayed result layers
    registration: Registration2D | None = None
       
""" 
CorrelationSession
│
├── flm
│   ├── original_image
│   ├── image
│   ├── original_points
│   ├── points
│   └── orientation_matrix
│
├── tem
│   ├── original_image
│   ├── image
│   ├── original_points
│   ├── points
│   └── orientation_matrix
│
├── registration
    
"""
    
