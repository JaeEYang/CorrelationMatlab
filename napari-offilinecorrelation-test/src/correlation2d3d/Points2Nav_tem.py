import numpy as np
import NavBuilt as nb
import SupportNav as sn

def coords_to_points(coord_file, nav_file, output_file):
    """
    Append new PointItems (from X Y Z coords in CSV) to an existing NAV file.
    Keeps all maps/items unchanged, adds new points associated with the first MapItem.
    """
    # --- Parse NAV file
    navdata = nb.parseNavFile(nav_file)

    if not navdata.Maps:
        raise ValueError("No MapItem found in provided NAV file")

    # Select the first MapItem to associate with
    first_map = navdata.Maps[0]
    print(f"Associating new points with MapID: {first_map.MapID}")

    # --- Load coordinates
    coords = np.loadtxt(coord_file, delimiter=",")
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)  # handle single-row case
    if coords.shape[1] != 3:
        raise ValueError("Coordinate file must have exactly 3 columns (X Y Z)")

    # --- Convert coords to PointItems
    new_points = []
    start_index = len(navdata.Points) + len(navdata.Maps) + 1  # continue Item numbering
    for i, (x, y, z) in enumerate(coords, start=start_index):
        p = nb.PointItem()
        p.Label = str(i)
        p.StageXYZ = [float(x), float(y), float(z)]
        p.PtsX = float(x)
        p.PtsY = float(y)
        p.DrawnID = first_map.MapID
        navdata.addPointItem(p)
        new_points.append(p)

    # --- Write out new NAV file
    with open(output_file, "w") as f:
        # Copy original template.nav content directly
        with open(nav_file, "r") as fin:
            f.write(fin.read())

        # Append only the new point items
        for p in new_points:
            f.write("\n\n")
            f.write("\n".join(p.getText()) + "\n")

    print(f"Nav file written to {output_file} with {len(new_points)} new points appended.")


# Example usage
if __name__ == "__main__":
    coord_file = input("Enter coordinate file (X Y Z per row): ")
    nav_file = input("Enter existing NAV file (template): ")
    output_file = input("Enter output NAV file name: ")

    coords_to_points(coord_file, nav_file, output_file)
