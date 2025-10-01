"""
Core transformation utilities for FLM-TEM correlation.
Converted from MATLAB scripts: computeTransform.m and transformPoints.m
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_transform(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 affine transformation matrix from FLM to TEM coordinates.
    
    Converted from MATLAB: M = Q * P' * inv(P * P')
    
    Parameters
    ----------
    P : np.ndarray
        3xN matrix of source points (FLM coordinates) in homogeneous coordinates.
        Rows are [x, y, 1] for each point.
    Q : np.ndarray
        3xN matrix of target points (TEM coordinates) in homogeneous coordinates.
        Rows are [x, y, 1] for each point.
    
    Returns
    -------
    transform_matrix : np.ndarray
        3x3 affine transformation matrix that maps P to Q.
    """
    # M = Q * P' * inv(P * P')
    transform_matrix = Q @ P.T @ np.linalg.inv(P @ P.T)
    return transform_matrix


def transform_points(transform_matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply transformation matrix to points.
    
    Converted from MATLAB: transformedPts = M * pts
    
    Parameters
    ----------
    transform_matrix : np.ndarray
        3x3 transformation matrix.
    pts : np.ndarray
        3xN matrix of points in homogeneous coordinates [x, y, 1].
    
    Returns
    -------
    transformed_pts : np.ndarray
        3xN matrix of transformed points.
    """
    transformed_pts = transform_matrix @ pts
    return transformed_pts


def load_registration_points(csv_path: str) -> np.ndarray:
    """
    Load points from CSV and convert to homogeneous coordinates (3xN format).
    CSVs have no header row - just comma-separated x,y,1 values.
    """
    # Read without header since files have no header row
    df = pd.read_csv(csv_path, header=None)
    
    # Use first two columns (x and y coordinates)
    points = df.iloc[:, :2].values
    
    # Convert to 3xN homogeneous coordinates
    N = points.shape[0]
    points_homogeneous = np.vstack([
        points[:, 0],  # x coordinates
        points[:, 1],  # y coordinates
        np.ones(N)     # homogeneous coordinate
    ])
    
    return points_homogeneous


def save_transformed_points(transformed_pts: np.ndarray, output_path: str):
    """
    Save transformed points to CSV file without header row.
    Saves all 3 columns: x, y, z.
    """
    # Transpose to get rows as points (N x 3 format)
    # transformed_pts is 3xN, we need Nx3 for CSV output
    output_data = transformed_pts.T  # Now Nx3
    
    # Save without index and without header
    df = pd.DataFrame(output_data)
    df.to_csv(output_path, index=False, header=False)
    print(f"Saved {transformed_pts.shape[1]} transformed points to {output_path}")


def compute_warped_bounds(img_shape: tuple, M: np.ndarray) -> tuple:
    """
    Compute the bounding box needed to contain the warped image.
    
    Parameters
    ----------
    img_shape : tuple
        (height, width) of input image
    M : np.ndarray
        3x3 transformation matrix
    
    Returns
    -------
    output_shape : tuple
        (height, width) for output image
    offset : np.ndarray
        Translation offset to apply [x_offset, y_offset]
    """
    h, w = img_shape[:2]
    
    # Define the four corners of the input image in homogeneous coordinates
    corners = np.array([
        [0, 0, 1],      # top-left
        [w, 0, 1],      # top-right
        [w, h, 1],      # bottom-right
        [0, h, 1]       # bottom-left
    ]).T  # 3x4
    
    # Transform corners
    transformed_corners = M @ corners  # 3x4
    
    # Extract x, y coordinates (ignore homogeneous coordinate)
    x_coords = transformed_corners[0, :]
    y_coords = transformed_corners[1, :]
    
    # Find bounding box
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Compute output size (round up to include all pixels)
    output_width = int(np.ceil(x_max - x_min))
    output_height = int(np.ceil(y_max - y_min))
    
    # Offset to shift the image so top-left corner is at (0, 0)
    offset = np.array([x_min, y_min])
    
    return (output_height, output_width), offset

def warp_image(img: np.ndarray, M: np.ndarray, ref_size: tuple = None) -> tuple:
    """
    Transform image using 3x3 affine transformation matrix.
    Matches MATLAB's imwarp behavior with imref2d.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (2D grayscale or 3D RGB).
    M : np.ndarray
        3x3 transformation matrix.
    ref_size : tuple, optional
        (height, width) of output reference image.
    
    Returns
    -------
    warped_img : np.ndarray
        Transformed image.
    offset : None
        Always None when using ref_size (for API compatibility)
    """
    from scipy.ndimage import affine_transform
    
    if ref_size is None:
        raise ValueError("ref_size must be provided")
    
    # Extract 2x2 linear part and translation from 3x3 matrix
    # M maps from input (FLM) to output (TEM) coordinates
    linear_part = M[:2, :2]
    translation = M[:2, 2]
    
    # scipy's affine_transform needs the INVERSE transformation
    # because it maps output pixel locations back to input locations
    try:
        inv_linear = np.linalg.inv(linear_part)
    except np.linalg.LinAlgError:
        # Singular matrix, return zeros
        if img.ndim == 3:
            return np.zeros((*ref_size, img.shape[2]), dtype=img.dtype), None
        return np.zeros(ref_size, dtype=img.dtype), None
    
    # Compute inverse translation
    # For each output pixel at (x_out, y_out), we need to find (x_in, y_in)
    # x_out = M @ x_in, so x_in = M^-1 @ x_out
    inv_translation = -inv_linear @ translation
    
    # MATLAB uses pixel-center coordinates (0.5, 0.5) for first pixel
    # scipy uses pixel-corner coordinates (0, 0) for first pixel
    # We need to adjust for this half-pixel offset
    
    # Adjust the inverse transformation to account for pixel-center convention
    # When sampling, shift by 0.5 to align with MATLAB's convention
    inv_translation = inv_translation + inv_linear @ np.array([0.5, 0.5]) - 0.5
    
    # Handle RGB images (3 channels)
    if img.ndim == 3:
        warped_img = np.zeros((*ref_size, img.shape[2]), dtype=img.dtype)
        for c in range(img.shape[2]):
            warped_img[:, :, c] = affine_transform(
                img[:, :, c],
                matrix=inv_linear,
                offset=inv_translation,
                output_shape=ref_size,
                order=1,  # Linear interpolation (bilinear)
                mode='constant',
                cval=0
            )
    else:
        # Grayscale image
        warped_img = affine_transform(
            img,
            matrix=inv_linear,
            offset=inv_translation,
            output_shape=ref_size,
            order=1,
            mode='constant',
            cval=0
        )
    
    return warped_img, None

def compare_with_expected(output_csv: str, expected_csv: str) -> Tuple[bool, str]:
    """
    Compare output CSV with expected test results.
    
    Parameters
    ----------
    output_csv : str
        Path to generated output CSV.
    expected_csv : str
        Path to expected results CSV.
    
    Returns
    -------
    matches : bool
        Whether the results match within tolerance.
    message : str
        Comparison details.
    """
    try:
        output_df = pd.read_csv(output_csv)
        expected_df = pd.read_csv(expected_csv)
        
        # Compare shapes
        if output_df.shape != expected_df.shape:
            return False, f"Shape mismatch: output {output_df.shape} vs expected {expected_df.shape}"
        
        # Compare values with tolerance
        output_vals = output_df.values
        expected_vals = expected_df.values
        
        max_diff = np.max(np.abs(output_vals - expected_vals))
        mean_diff = np.mean(np.abs(output_vals - expected_vals))
        
        tolerance = 1e-6
        matches = max_diff < tolerance
        
        message = f"Max difference: {max_diff:.2e}\nMean difference: {mean_diff:.2e}"
        if matches:
            message = "✓ Results match expected output!\n" + message
        else:
            message = "✗ Results differ from expected output.\n" + message
        
        return matches, message
        
    except Exception as e:
        return False, f"Error comparing files: {str(e)}"
