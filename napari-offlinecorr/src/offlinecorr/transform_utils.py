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
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing x,y coordinates.
    
    Returns
    -------
    points_homogeneous : np.ndarray
        3xN array with rows [x, y, 1].
    """
    df = pd.read_csv(csv_path)
    
    # Try to find x,y columns (case insensitive)
    cols = df.columns.str.lower()
    if 'x' in cols.values and 'y' in cols.values:
        x_col = df.columns[cols == 'x'][0]
        y_col = df.columns[cols == 'y'][0]
        points = df[[x_col, y_col]].values
    else:
        # Use first two columns
        points = df.iloc[:, :2].values
    
    # Convert to 3xN homogeneous coordinates
    # points is Nx2, we need 3xN
    N = points.shape[0]
    points_homogeneous = np.vstack([
        points[:, 0],  # x coordinates
        points[:, 1],  # y coordinates
        np.ones(N)     # homogeneous coordinate
    ])
    
    return points_homogeneous


def save_transformed_points(transformed_pts: np.ndarray, output_path: str):
    """
    Save transformed points to CSV file.
    
    Parameters
    ----------
    transformed_pts : np.ndarray
        3xN matrix of transformed points in homogeneous coordinates.
    output_path : str
        Path for output CSV file.
    """
    # Extract x, y from homogeneous coordinates (first two rows)
    df = pd.DataFrame({
        'x': transformed_pts[0, :],
        'y': transformed_pts[1, :]
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {transformed_pts.shape[1]} transformed points to {output_path}")


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
