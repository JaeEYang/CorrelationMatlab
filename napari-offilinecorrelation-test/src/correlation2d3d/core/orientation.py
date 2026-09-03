import numpy as np
from correlation2d3d.core.geometry import   Points2D

# gotta make this a class with diffren attributes and for each 


def flip_horizontal(image:np.ndarray, points:Points2D) -> tuple[np.ndarray, Points2D]:
    
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
    
    flipped_xy = points.xy.copy() # get the landmarks save them 
    # flip the landmarks aswell and modify only the x values as y would stay the same for left right flipping 
    flipped_xy[:, 0] = (
        width - 1 - flipped_xy[:, 0]
    )

    return (
        flipped_image,
        Points2D(flipped_xy),
    )