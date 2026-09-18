import numpy as np

from correlation2d3d.offline_correlation import (
    _build_multiscale_pyramid,
)


def test_multiscale_level_zero_preserves_full_scientific_image():
    image = np.arange(
        2048 * 2048,
        dtype=np.uint16,
    ).reshape(2048, 2048)

    pyramid = _build_multiscale_pyramid(image)

    assert len(pyramid) >= 2

    # Level 0 must be the untouched full-resolution scientific data.
    assert pyramid[0] is image

    np.testing.assert_array_equal(
        pyramid[0],
        image,
    )

    assert pyramid[0].shape == (2048, 2048)

    np.testing.assert_array_equal(
        pyramid[1],
        image[::2, ::2],
    )