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
    
def test_to_rc_swaps_xy_to_row_column():
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    rc = points.to_rc()

    np.testing.assert_array_equal(
        rc,
        np.array([
            [20, 10],
            [40, 30],
        ]),
    )
    
def test_to_rc_returns_independent_storage():
    points = Points2D([[10, 20]])

    rc = points.to_rc()

    assert not np.shares_memory(points.xy, rc)

    rc[0, 0] = 999

    assert points.xy[0, 1] == 20
    
def test_from_rc_swaps_row_column_to_xy():
    points = Points2D.from_rc([
        [20, 10],
        [40, 30],
    ])

    np.testing.assert_array_equal(
        points.xy,
        np.array([
            [10, 20],
            [30, 40],
        ]),
    )
    
def test_rc_round_trip_preserves_coordinates():
    original = Points2D([
        [10.5, 20.25],
        [30.75, 40.125],
    ])

    restored = Points2D.from_rc(original.to_rc())

    np.testing.assert_array_equal(
        restored.xy,
        original.xy,
    )
    
def test_subset_selects_points_with_boolean_mask():
    points = Points2D([
        [10, 20],
        [30, 40],
        [50, 60],
    ])

    selected = points.subset(
        np.array([True, False, True])
    )

    np.testing.assert_array_equal(
        selected.xy,
        np.array([
            [10, 20],
            [50, 60],
        ]),
    )
    
def test_subset_rejects_non_boolean_mask():
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    with pytest.raises(TypeError):
        points.subset([1, 0])
        
def test_subset_rejects_two_dimensional_mask():
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    with pytest.raises(ValueError):
        points.subset([[True, False]])
        
def test_subset_rejects_wrong_length_mask():
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    with pytest.raises(ValueError):
        points.subset([True])
        
        
def test_subset_can_select_no_points():
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    selected = points.subset(
        np.array([False, False])
    )

    assert selected.xy.shape == (0, 2)
    assert len(selected) == 0