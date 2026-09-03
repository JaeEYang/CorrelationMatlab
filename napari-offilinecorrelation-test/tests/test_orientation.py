import numpy as np

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.orientation import flip_horizontal


def test_flip_horizontal_moves_image_and_points_together():
    image = np.zeros((3, 4), dtype=np.float64)

    image[1, 0] = 1.0

    points = Points2D([
        [0, 1],
        [3, 2],
    ])

    flipped_image, flipped_points = flip_horizontal(
        image,
        points,
    )

    expected_image = np.zeros((3, 4), dtype=np.float64)
    expected_image[1, 3] = 1.0

    expected_points = np.array([
        [3, 1],
        [0, 2],
    ])

    np.testing.assert_array_equal(
        flipped_image,
        expected_image,
    )

    np.testing.assert_allclose(
        flipped_points.xy,
        expected_points,
    )