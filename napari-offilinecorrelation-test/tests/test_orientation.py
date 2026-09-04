import numpy as np

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.orientation import flip_horizontal, horizontal_flip_matrix, apply_orientation_to_points,flip_vertical,vertical_flip_matrix


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
    
    
def test_flip_horizontal_does_not_require_points():
    image = np.arange(12).reshape(3, 4)

    flipped_image, flipped_points = flip_horizontal(
        image
    )

    np.testing.assert_array_equal(
        flipped_image,
        np.fliplr(image),
    )

    assert flipped_points is None
    
def test_late_loaded_points_follow_horizontal_flip():
    width = 4

    orientation = np.eye(
        3,
        dtype=np.float64,
    )

    orientation = (
        horizontal_flip_matrix(width)
        @ orientation
    )

    points_loaded_later = Points2D([
        [0.0, 1.0],
        [3.0, 2.0],
    ])

    current_points = apply_orientation_to_points(
        points_loaded_later,
        orientation,
    )

    expected = np.array([
        [3.0, 1.0],
        [0.0, 2.0],
    ])

    np.testing.assert_allclose(
        current_points.xy,
        expected,
    )
def test_flip_vertical_moves_image_and_points_together():
    image = np.zeros((3, 4), dtype=np.float64)

    image[0, 1] = 1.0

    points = Points2D([
        [1, 0],
        [3, 2],
    ])

    flipped_image, flipped_points = flip_vertical(
        image,
        points,
    )

    expected_image = np.zeros((3, 4), dtype=np.float64)
    expected_image[2, 1] = 1.0

    expected_points = np.array([
        [1, 2],
        [3, 0],
    ])

    np.testing.assert_array_equal(
        flipped_image,
        expected_image,
    )

    np.testing.assert_allclose(
        flipped_points.xy,
        expected_points,
    )
    
def test_flip_vertical_does_not_require_points():
    image = np.arange(12).reshape(3, 4)

    flipped_image, flipped_points = flip_vertical(
        image
    )

    np.testing.assert_array_equal(
        flipped_image,
        np.flipud(image),
    )

    assert flipped_points is None
    
# loading points after image has already been flipped it should still apply the correct tranformation
def test_late_loaded_points_follow_vertical_flip():
    height = 3

    orientation = np.eye(
        3,
        dtype=np.float64,
    )

    orientation = (
        vertical_flip_matrix(height)
        @ orientation
    )

    points_loaded_later = Points2D([
        [1.0, 0.0],
        [3.0, 2.0],
    ])

    current_points = apply_orientation_to_points(
        points_loaded_later,
        orientation,
    )

    expected = np.array([
        [1.0, 2.0],
        [3.0, 0.0],
    ])

    np.testing.assert_allclose(
        current_points.xy,
        expected,
    )