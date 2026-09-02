from dataclasses import dataclass
import numpy as np

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.transform import Registration2D


'''        CorrelationSession
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
   Points2D    Registration2D    images
 immutable       immutable       arrays

        session itself is mutable '''
        
# this class is used to store the state of a correlation session, 
# including the images, points, and registration information. 
# this will me mutable because the state is expected to change as the user interacts with the application, e.g. loading images, selecting points, and computing registrations.
# this becomes home for the changing state of the correlation session.

@dataclass
class CorrelationSession:
    flm_image: np.ndarray | None = None
    tem_image: np.ndarray | None = None

    flm_points: Points2D | None = None
    tem_points: Points2D | None = None

    registration: Registration2D | None = None
    warped_flm: np.ndarray | None = None