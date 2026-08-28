import numpy as np

# -----------------------------
# Test Map Item (from your NAV)
# -----------------------------
map_item = {
    "MapScaleMat": np.array([[10, 0],
                             [0, -10]], dtype=float),
    "MapWidthHeight": np.array([400, 462], dtype=float),
    "RawStageXY": np.array([0.0, 0.0], dtype=float),  # from StageXYZ in NAV
}

# -----------------------------
# Pixel-to-Stage Function (debugging)
# -----------------------------
def pixel_to_stage_debug(map_item, x_pix: float, y_pix: float) -> np.ndarray:
    """
    Convert pixel coords (X, Y) to stage coords using SerialEM convention.
    """
    mat_nm = np.array(map_item["MapScaleMat"], dtype=float)
    # MapScaleMat is in nm/pixel. Convert to um/pixel.
    mat_um = mat_nm * 0.01   # adjust factor if needed (0.01 for nm->um)

    center = map_item["MapWidthHeight"] / 2.0
    pix_vec = np.array([x_pix, y_pix], dtype=float) - center

    stage_xy = np.dot(mat_um, pix_vec) + map_item["RawStageXY"]
    return stage_xy


# -----------------------------
# Run the test
# -----------------------------
def run_test():
    # Reversed X/Y pairs in pixel coordinates
    test_points = [
        (50,  248),
        (224, 284),
        (49,  306),
        (146, 352),
    ]

    # Expected stage X, Y (in microns)
    expected_stage = [
        (-14.8, -1.6),
        (  2.4, -4.9),
        (-15.2, -7.5),
        (-5.4, -11.9),
    ]

    print("==== Pixel-to-Stage Transformation Test ====\n")
    for i, ((x_pix, y_pix), (sx_exp, sy_exp)) in enumerate(zip(test_points, expected_stage), start=1):
        stage_xy = pixel_to_stage_debug(map_item, x_pix, y_pix)
        sx_calc, sy_calc = stage_xy

        print(f"[Point {i}] Pixel (X={x_pix}, Y={y_pix})")
        print(f"  -> Computed stage (X={sx_calc:.2f}, Y={sy_calc:.2f})")
        print(f"  -> Expected stage (X={sx_exp:.2f}, Y={sy_exp:.2f})")
        print(f"  -> delta error (X={sx_calc - sx_exp:.2f}, Y={sy_calc - sy_exp:.2f})\n")


if __name__ == "__main__":
    run_test()
