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
                    │
                 warped FLM
      
# One place to answer what is the CURRENT STATE OF EVERYTHING ? this becomes home for the changing states.
        
      
        
     Mutable state for one imaging modality.

    original_image:
        Pixel data exactly as loaded.

    image:
        Current working orientation of the image.

    original_points:
        Landmark coordinates in the original image coordinate system.

    points:
        Landmark coordinates in the current working coordinate system.

    orientation_matrix:
        Maps original image coordinates into the current working coordinates.
    """
@dataclass
class ModalityState:
 
    original_image: np.ndarray | None = None 
    image: np.ndarray | None = None 
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
    registration: Registration2D | None = None
    warped_flm: np.ndarray | None = None
    
    
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
│
└── warped_flm
    
"""
    