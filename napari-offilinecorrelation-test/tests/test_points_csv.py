import numpy as np
import pytest
from pathlib import Path

from correlation2d3d.fileio.points_csv import (
    read_points_csv,
    write_points_csv,
)

from correlation2d3d.core.geometry import Points2D


def test_read_points_csv_reads_xy_coordinates(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "10,20\n"
        "30,40\n",
        encoding="utf-8",
    )

    points = read_points_csv(path, order="xy")

    np.testing.assert_array_equal(
        points.xy,
        np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
    )


def test_read_points_csv_converts_yx_to_xy(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "20,10\n"
        "40,30\n",
        encoding="utf-8",
    )

    points = read_points_csv(path, order="yx")

    np.testing.assert_array_equal(
        points.xy,
        np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
    )


def test_read_points_csv_accepts_homogeneous_column(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "10,20,1\n"
        "30,40,1\n",
        encoding="utf-8",
    )

    points = read_points_csv(path)

    np.testing.assert_array_equal(
        points.xy,
        np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ]),
    )


def test_read_points_csv_rejects_nonunit_homogeneous_column(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "10,20,1\n"
        "30,40,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_points_csv(path)


def test_read_points_csv_rejects_unknown_order(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "10,20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_points_csv(path, order="banana")

def test_read_points_csv_rejects_mixed_column_counts(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text(
        "10,20\n"
        "30,40,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_points_csv(path)
        
def test_read_points_csv_reads_repository_reference_data():
    repo_root = Path(__file__).resolve().parents[2] # get the correlationMatlab repository root directory
    path = repo_root / "data" / "Item2_X7Y6_FLM_RegSpread9.csv"

    points = read_points_csv(path)

    assert points.xy.shape == (9, 2)

    np.testing.assert_allclose(
        points.xy[0],
        np.array([321.0, 248.5]),
    )
    
# write test for write_points_csv


def test_write_points_csv_writes_xy_coordinates(tmp_path):
    path = tmp_path / "points.csv"
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    write_points_csv(path, points, order="xy")

    assert path.read_text(encoding="utf-8").splitlines() == [
        "10.0,20.0",
        "30.0,40.0",
    ]
    
def test_write_points_csv_writes_yx_coordinates(tmp_path):
    path = tmp_path / "points.csv"
    points = Points2D([
        [10, 20],
        [30, 40],
    ])

    write_points_csv(path, points, order="yx")

    assert path.read_text(encoding="utf-8").splitlines() == [
        "20.0,10.0",
        "40.0,30.0",
    ]
    
def test_points_csv_round_trip_preserves_points(tmp_path):
    path = tmp_path / "points.csv"

    original = Points2D([
        [10.25, 20.5],
        [30.75, 40.125],
    ])

    write_points_csv(path, original)
    restored = read_points_csv(path)

    np.testing.assert_array_equal(
        restored.xy,
        original.xy,
    )