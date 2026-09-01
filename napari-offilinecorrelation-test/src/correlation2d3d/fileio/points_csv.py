import csv
from typing import Literal
from pathlib import Path

import numpy as np

from correlation2d3d.core.geometry import Points2D

Order = Literal["xy", "yx"]  # Define a type for the order parameter


'''
file text
   ↓
csv.reader
   ↓
validate file structure
   ↓
convert strings to floats
   ↓
remove homogeneous 1 if present
   ↓
convert yx → xy if requested
   ↓
Points2D
'''
def read_points_csv(path: str | Path,*,order: Order = "xy") -> Points2D: # when called order must be specified explicitly, e.g read_points_csv(path, order="xy") 
    """
    Read a CSV file containing 2D points and return them as a Points2D object.

    Parameters:
        path (str): The path to the CSV file.
        order (Order, optional): The order of the coordinates in the CSV file. Defaults to "xy".

    Returns:
        Points2D: A Points2D object containing the read points.
    
    """
    if order not in ("xy", "yx"):
        raise ValueError(
            f"order must be 'xy' or 'yx', got {order!r}"
        )
    
    rows = []
    expected_columns = None
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start = 1): # 
            if not row or all(not cell.strip() for cell in row):  # Skip empty rows or rows with only whitespace
                continue
            if len(row) not in (2,3):  # Check if the row has 2 or 3 columns
                raise ValueError(
                    f"row {line_number} must contain 2 or 3 columns, "
                    f"got {len(row)}"
                )
                
            # define the expected number of columns based on the first non-empty row
            # either there are 3 columns and the last one is 1, or there are 2 columns. All rows must match this.
            if expected_columns is None:
                expected_columns = len(row)
            elif len(row) != expected_columns:
                raise ValueError(
                    f"row {line_number} has {len(row)} columns, "
                    f"expected {expected_columns}"
                    ) 
            #make sure all values are numeric and convert them to float. and raise value error is not a number              
            try:
                values = [float(cell) for cell in row]  # Convert all fields to floats
            except ValueError as exc:
                raise ValueError(
                     f"row {line_number} contains a non-numeric value"
                ) from exc
            
            # if there is a third column it must be 1, else raise an error
            if len(values) == 3:
                if values[2] != 1.0:
                    raise ValueError(
                        f"row {line_number} has a third column "
                        "that is not the homogeneous value 1"
                    )

                values = values[:2] # remove the ones column, so we can store them as 2D points in Points2D
                
            rows.append(values)
    if not rows:
        array = np.empty((0, 2), dtype=np.float64) # if empty we need the correct shape
    else:
        array = np.asarray(rows, dtype=np.float64) # else save it as array

    if order == "yx": # flip the order to core consistent xy 
        array = array[:, ::-1]

    return Points2D(array) # return the Points2D object with the read points

'''Points2D
canonical (x,y)
     |
     | if order="yx", swap columns
     v
CSV representation
     |
     v
write file'''

def write_points_csv(path: str | Path, points: Points2D, *, order: Order = "xy") -> None:
    if order not in ("xy", "yx"):
        raise ValueError(
            f"order must be 'xy' or 'yx', got {order!r}"
        )
    array = points.xy
    if order == "yx":
        array = array[:,::-1] # Flip the order to yx if specified
        
    # now we open the csv and write to it. We will write the points as rows, with each point as a row of two columns.
    with Path(path).open("w", newline ="", encoding="utf-8") as file:
        writer = csv.writer(file) # write the points to the csv file
        writer.writerows(array) # write the points to the csv file
        
       