import csv
import NavBuilt as nb
import SupportNav as sn

def two_value(value):  ## for PtsX PtsY
    values = value.split(",")
    if len(values) != 2:
        raise ValueError("Expected two comma-separated values")
    return values

def four_value(value):  ## for MapScaleMat
    values = value.split(",")
    if len(values) != 4:
        raise ValueError("Expected four comma-separated values")
    return values

def createPrototypeItem(nav):  ## generate a prototype
    for item in nav:
        if item[0][1].isnumeric():
            itemContents = []
            for line in item:
                if line[0] in ["PtsX", "PtsY", "Item"]:
                    continue
                itemContents.append(line)
            return itemContents
    return []

def csv_to_nav(csv_file, nav_template, output_file,
               mapwidth, mapheight, mapscalemat,
               colorId, regisId, drawnId,
               imported=-1, backlash=(0, 0), start_index=1):
    print(f"Input CSV: {csv_file}")
    print(f"Template NAV: {nav_template}")
    print(f"Output NAV: {output_file}")

    with open(csv_file, "r") as f:
        csvLines = list(csv.reader(f, delimiter=","))
    xyArray = []
    for line in csvLines:
        try:
            x = float(line[0]) - mapwidth / 2
            y = float(line[1]) - mapheight / 2
            xyArray.append((x, y))
        except Exception:
            pass

    print("First 3 CSV coords:", xyArray[:3])

    GNnav = sn.parseFile(nav_template)

    msn = [[float(mapscalemat[0]), float(mapscalemat[1])],
           [float(mapscalemat[2]), float(mapscalemat[3])]]
    
    msnInv = sn.invertMatrix(msn)

    XYArray = sn.transposeMatrix(xyArray)
    XYTransform = sn.matrixMultiply(msnInv, XYArray)

    prototype = createPrototypeItem(GNnav)

    with open(output_file, "a") as f:
        f.write(sn.createNavItem(GNnav[0]))
        ## write Map item
        MapItemBlob = sn.createNavItem(GNnav[1])
        lines = MapItemBlob.split("\n")
        for line in lines:
            if "Imported" in line:
                f.write(f"Imported = {imported}\n")
            elif "MapWidthHeight" in line:
                f.write(f"MapWidthHeight = {mapwidth} {mapheight}\n")
            elif "MapScaleMat" in line:
                f.write(f"MapScaleMat = {msn[0][0]} {msn[0][1]} {msn[1][0]} {msn[1][1]}\n")
            else:
                f.write(line + "\n")

        ## write point items
        itemIndex = start_index
        for x, y in zip(XYTransform[0], XYTransform[1]):
            newItem = [["Item", itemIndex]]
            itemIndex += 1

            for line in prototype:
                if line[0] == "StageXYZ":
                    StageXYZ = "{:.3f} {:.3f} {:.3f}".format(x, y, 0)
                    newItem.append(["StageXYZ", StageXYZ])
                elif line[0] == "Color":
                    newItem.append(["Color", colorId])
                elif line[0] == "Regis":
                    newItem.append(["Regis", regisId])
                elif line[0] == "Imported":
                    newItem.append(["Imported", imported])
                else:
                    newItem.append(line)
            newItem.append(["BklshXY", f"{backlash[0]} {backlash[1]}"])
            newItem.append(["PtsX", "{:.3f}".format(x)])
            newItem.append(["PtsY", "{:.3f}".format(y)])
            newItem.append(["DrawnID", drawnId])

            outString = sn.createNavItem(newItem)
            f.write(outString)
    print(f"Nav file generated: {output_file}")

if __name__ == "__main__":
    csv_file = input("Enter CSV file path: ")
    nav_template = input("Enter template NAV file path: ")
    output_file = input("Enter output NAV filename: ")
    mapwidth = int(input("Map width (pixels): "))
    mapheight = int(input("Map height (pixels): "))
    mapscalemat = four_value(input("Map scale matrix (4 comma-separated numbers): "))
    colorId = input("Color ID: ")
    regisId = input("Regis ID: ")
    drawnId = input("Drawn ID: ")

    csv_to_nav(csv_file, nav_template, output_file,
               mapwidth, mapheight, mapscalemat,
               colorId, regisId, drawnId)
    

           


