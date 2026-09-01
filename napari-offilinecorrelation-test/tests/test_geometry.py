import numpy as np
import pytest

from correlation2d3d.core.geometry import Points2D


def test_points2d_normalizes_numeric_input_to_float64():
    points = Points2D([[1, 2], [3, 4]])

    assert points.xy.shape == (2, 2)
    assert points.xy.dtype == np.float64
    np.testing.assert_array_equal(
        points.xy,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_points2d_copies_input_array():
    source = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float64,
    )

    points = Points2D(source)

    assert not np.shares_memory(source, points.xy)


def test_modifying_source_after_construction_does_not_change_points():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])

    points = Points2D(source)

    source[0, 0] = 999.0

    assert points.xy[0, 0] == 1.0


def test_points2d_storage_is_read_only():
    points = Points2D([[1, 2], [3, 4]])

    assert not points.xy.flags.writeable

    with pytest.raises(ValueError):
        points.xy[0, 0] = 999.0


def test_points2d_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        Points2D([1, 2])


def test_points2d_rejects_wrong_number_of_columns():
    with pytest.raises(ValueError):
        Points2D([[1, 2, 3], [4, 5, 6]])


def test_points2d_rejects_nan():
    with pytest.raises(ValueError):
        Points2D([[1.0, np.nan]])


def test_points2d_rejects_infinity():
    with pytest.raises(ValueError):
        Points2D([[1.0, np.inf]])


def test_points2d_accepts_empty_collection():
    points = Points2D(np.empty((0, 2)))

    assert points.xy.shape == (0, 2)
    assert len(points) == 0


def test_points2d_len_returns_number_of_points():
    points = Points2D(
        [[1, 2], [3, 4], [5, 6]]
    )

    assert len(points) == 3