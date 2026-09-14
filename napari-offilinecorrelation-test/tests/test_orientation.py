import numpy as np

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.transform import apply_affine_matrix
from correlation2d3d.core.orientation import (
    horizontal_flip_matrix,
    vertical_flip_matrix,
    rotation_matrix,
)


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

    current_points = apply_affine_matrix(
        orientation,
        points_loaded_later,
    )

    expected = np.array([
        [3.0, 1.0],
        [0.0, 2.0],
    ])

    np.testing.assert_allclose(
        current_points.xy,
        expected,
    )


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

    current_points = apply_affine_matrix(
        orientation,
        points_loaded_later,
    )

    expected = np.array([
        [1.0, 2.0],
        [3.0, 0.0],
    ])

    np.testing.assert_allclose(
        current_points.xy,
        expected,
    )


def test_rotation_matrix_rotates_point_counterclockwise_about_center():
    matrix = rotation_matrix(
        height=7,
        width=7,
        angle_degrees=90.0,
    )

    point = Points2D([
        [5.0, 3.0],
    ])

    rotated = apply_affine_matrix(
        matrix,
        point,
    )

    np.testing.assert_allclose(
        rotated.xy,
        np.array([
            [3.0, 1.0],
        ]),
        atol=1e-12,
    )
