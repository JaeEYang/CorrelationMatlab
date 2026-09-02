import numpy as np

from correlation2d3d.core.transform import Registration2D
from correlation2d3d.core.warp import warp_image

def test_warp_image_applies_translation():
    #5x5 image with a single pixel set to 1.0 at (1, 1)--> core convention remember (x,y) = (col,row)
    image = np.zeros((5, 5), dtype=np.float64)
    image[1, 1] = 1.0

    registration = Registration2D(
        matrix=np.array([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
    )

    warped = warp_image( image, registration, output_shape=(5, 5), order=0,)
    
    expected = np.zeros((5, 5), dtype=np.float64)
    expected[2, 3] = 1.0

    np.testing.assert_allclose(
        warped,
        expected,
        atol=1e-12,
    )

# this test checks that the warp_image function correctly
# handles RGB images by preserving the color channels during the warping process.
# It creates a 5x5 RGB image with a single pixel set to a specific color, 
# applies a translation transformation, and verifies that the warped image has the expected color at the new location.
def test_warp_image_preserves_rgb_channels():
    image = np.zeros((5, 5, 3), dtype=np.float64)

    image[1, 1] = [1.0, 2.0, 3.0]

    registration = Registration2D(
        matrix=np.array([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
    )

    warped = warp_image(
        image,
        registration,
        output_shape=(5, 5),
        order=0,
    )

    assert warped.shape == (5, 5, 3)

    np.testing.assert_allclose(
        warped[2, 3],
        [1.0, 2.0, 3.0],
        atol=1e-12,
    )