import numpy as np
import pytest

from correlation2d3d.core.geometry import Points2D
from correlation2d3d.core.transform import fit_affine
from correlation2d3d.core.transform import Registration2D

def test_fit_affine_recovers_exact_transform():
    source = Points2D([
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    destination = Points2D([
        [5, 10],
        [7, 10],
        [5, 13],
    ])

    registration = fit_affine(
        source,
        destination,
    )

    expected = np.array([
        [2.0, 0.0, 5.0],
        [0.0, 3.0, 10.0],
        [0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        registration.matrix,
        expected,
        atol=1e-12,
    )
    
    
def test_registration_apply_transforms_points():
    registration = Registration2D(
        matrix=np.array([
            [2.0, 0.0, 5.0],
            [0.0, 3.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
    )

    points = Points2D([
        [2, 4],
        [-1, 3],
    ])

    transformed = registration.apply(points)

    expected = np.array([
        [9.0, 22.0],
        [3.0, 19.0],
    ])

    np.testing.assert_allclose(
        transformed.xy,
        expected,
        atol=1e-12,
    )

'''this is very useful basically: 
 can fit_affine() handle more than the minimum 3 correspondences ? 
 is np.linalg.lstsq() finding a best-fit affine transform when the data are slightly inconsistent ?
 does the fitted matrix stays close to the known underlying transform we constructed? 
 are the residuals are not all zero when one landmark is noisy? 
rmse is therefore positive, which is what we expect for imperfect landmark data.
'''
def test_fit_affine_handles_overdetermined_noisy_points():
    source = Points2D([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    destination = Points2D([
        [5.0, 10.0],
        [7.0, 9.75],
        [5.5, 13.0],
        [7.7, 12.65],  # slightly noisy measurement instead of real 7.5,12.75
    ])

    registration = fit_affine(
        source,
        destination,
    )

    expected_matrix = np.array([
        [2.0, 0.5, 5.0],
        [-0.25, 3.0, 10.0],
        [0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        registration.matrix,
        expected_matrix,
        atol=0.11,
    )

    assert registration.rmse is not None
    assert registration.rmse > 0.0
    
def test_fit_affine_rejects_mismatched_point_counts():
    source = Points2D([
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    destination = Points2D([
        [5, 10],
        [7, 10],
    ])

    with pytest.raises(ValueError):
        fit_affine(source, destination)

def test_fit_affine_rejects_collinear_source_points():
    source = Points2D([
        [0, 0],
        [1, 1],
        [2, 2],
    ])

    destination = Points2D([
        [5, 10],
        [7, 12],
        [9, 14],
    ])

    with pytest.raises(ValueError):
        fit_affine(source, destination)