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