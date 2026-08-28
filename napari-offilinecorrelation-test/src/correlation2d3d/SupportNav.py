##  A generic parser to expand upon Maps and Points

import sys
import os
import numpy as np

def getFilePath():
    ## Return the filepath to the folder which contains .csv file
    filepath = sys.path[0]
    for _ in range(3):
        filepath = os.path.dirname(filepath)
    return filepath

def parseFile(fileName):   
    ## Parse a .nav file into a nested list structure, nav[item][line][(key, value)]  
    with open(fileName, "r") as f:
        lines = f.readlines()

    itemList, currentItem = [], []
    for line in lines:
        line = line.strip().replace("[", "").replace("]", "")
        if not line:
            continue
        if "Item =" in line:
            if currentItem:
                itemList.append(currentItem)
            currentItem = []
        currentItem.append([s.strip() for s in line.split("=", 1)])

    if currentItem:
        itemList.append(currentItem)

    return itemList

def getFileWithKeyword(filepath, keyword):
    ## Return file with keyword in name, only if exactly one exists
    matches = [f for f in os.listdir(filepath) if keyword in f]
    if not matches:
        print(f"No files in {filepath} with keyword '{keyword}'")
        return None
    if len(matches) > 1:
        print(f"Too many files in {filepath} with keyword '{keyword}'")
        return None
    return matches[0]

def checkIfFileExists(fileName):
    ## Return True if file exists or cannot be opened safely, else False
    try:
        with open(fileName, "r"):
            print("Output file already exists")
        return True
    except IOError:
        return False
    except Exception as e:
        print("Unexpected error:", e)
        return True

def findMapScaleMat(nav):
    ## Return the first MapScaleMat found as a 2D numpy array
    for item in nav:
        for key, value in item:
            if key == "MapScaleMat":
                nums = [float(x) for x in value.split()]
                return np.array(nums).reshape(2, 2)
    return None

def createNavItem(item):
    ## Rebuild a nav item back into text format
    lines = []
    for key, value in item:
        if key == "Item":
            lines.append(f"\n[{key} = {value}]\n")
        else:
            lines.append(f"{key} = {value}\n")
    return "".join(lines)

def invertMatrix(mat):
    ## Return the inverse of a matrix using numpy
    mat = np.array(mat, dtype=float)
    return np.linalg.inv(mat)

def transposeMatrix(mat):
    return np.transpose(np.array(mat, dtype=float))

def matrixMultiply(mat1, mat2):
   ## Transform coordinate
    return np.dot(np.array(mat1, dtype=float), np.array(mat2, dtype=float))

def computeMsFLM(flm_points: np.ndarray, tem_points: np.ndarray) -> np.ndarray:
    """
    Compute the affine transform MsFLM (3x3) that maps FLM pixel coords to TEM pixel coords.
    Requires >=3 corresponding points.
    """
    if flm_points.shape != tem_points.shape or flm_points.shape[0] < 3:
        raise ValueError("FLM and TEM points must have the same shape and at least 3 points")
    flm_h = np.concatenate(
        (flm_points, np.ones((flm_points.shape[0], 1), dtype=float)),
        axis = 1
    )
    tem_h = np.concatenate(
        (tem_points, np.ones((tem_points.shape[0], 1), dtype=float)),
        axis = 1
    )
    MsFLM, *_ = np.linalg.lstsq(flm_h, tem_h, rcond=None)
    return MsFLM

def pixel_to_stage_o(map_item, x_pix: float, y_pix: float) -> np.ndarray:
    """
    Convert pixel coordinates (x_pix, y_pix) to stage XY positions using SerialEM convention.
    
    This uses:
      - MapScaleMat: nm/pixel scale & rotation matrix
      - MapWidthHeight: image dimensions in pixels (used to find image center)
      - RawStageXY: reference stage position of the map center

    The formula follows SerialEM convention:
        stage_xy = RawStageXY + (MapScaleMat_in_um x ( [x_pix, y_pix] - image_center ))
    """
    # 2x2 MapScaleMat in nm/pixel
    mat_nm = np.array(map_item.MapScaleMat, dtype=float)

    # Convert to um/pixel
    mat_um = mat_nm * 0.01

    # Image center
    center = np.array(map_item.MapWidthHeight, dtype=float) / 2.0

    # Pixel offset relative to center
    pix_vec = np.array([x_pix, y_pix], dtype=float) - center

    # Transform to stage coordinates
    stage_xy = np.dot(mat_um, pix_vec) + np.array(map_item.RawStageXY, dtype=float)
    return stage_xy

def pixel_to_stage(map_item, x_pix: float, y_pix: float) -> np.ndarray:
    M = np.array(map_item.MapScaleMat, dtype=float)
    M_inv = np.linalg.inv(M)
    WH = map_item.MapWidthHeight
    center = np.array([WH[0] * 0.5, WH[1] * 0.5], dtype=float)
    y_cart = WH[1] - y_pix

    pix_vec = np.array([x_pix, y_cart], dtype=float) - center

    stage_xy = M_inv @ pix_vec + np.array(map_item.StageXYZ[:2], dtype=float)

    return stage_xy
                                          






