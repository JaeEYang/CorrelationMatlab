import numpy as np
from skimage.transform import warp, AffineTransform
from correlation2d3d.core.transform import Registration2D

# define a function to warp an image using a given registration (affine transformation)
def warp_image(image:np.ndarray, registration:Registration2D, output_shape:tuple[int,int],*, order:int=1) -> np.ndarray:
    """
    Warp an image using a given registration (affine transformation).

    Parameters
    ----------
    image : np.ndarray
        The input image to be warped.
    registration : Registration2D
        The registration object containing the affine transformation matrix.
    output_shape : tuple[int, int]
        The shape of the output warped image (height, width).
    order : int, optional
        The order of the spline interpolation used for warping. Default is 1 (bilinear).

    Returns
    -------
    np.ndarray
        The warped image.
    """
    
    # Create an AffineTransform object from the registration matrix
    transform = AffineTransform(matrix=registration.matrix)
    
    # Use skimage's warp function to apply the transformation to the image, this does inverse mapping, so we use the inverse of the transform 
    # (if done forward, the image would be sampled at non-integer pixel locations, create holes, and not fill the output image properly)
    #order 0 is nearest neighbor, 1 is bilinear interpolation...
    warped_image = warp(
        image,
        inverse_map=transform.inverse,
        output_shape=output_shape,
        order=order,
        mode='constant', # outside the boundaries of the input image, fill with a constant value (cval)
        cval=0.0,
        preserve_range=True,# preserve the original pixel range of values in the input image (don't normalize to [0, 1])
    ) 
    
    return warped_image