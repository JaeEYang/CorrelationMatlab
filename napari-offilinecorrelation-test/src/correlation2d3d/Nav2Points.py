import csv
import NavBuilt as nb
import SupportNav as sn

def nav_to_csv(nav_file, csv_file):
    data = nb.parseNavFile(nav_file)
    outlist = []

    for m in data.Maps:
        pointcount = len(m.Points)
        print(f"Adding {pointcount} points from Map {m.MapID}")

        msm = m.MapScaleMat
        if m.MapWidthHeight:
            half_width = float(m.MapWidthHeight[0]) / 2.0
            half_height = float(m.MapWidthHeight[1]) / 2.0
        else:
            half_width = 0.0
            half_height = 0.0
        for point in m.Points:
            print(f" Point {point.Label}: PtsX={point.PtsX}, PtsY={point.PtsY}")
            XYArray = [[point.PtsX], [point.PtsY]]
            XYTransform = sn.matrixMultiply(msm, XYArray)
            X = XYTransform[0][0] + half_width
            Y = XYTransform[1][0] + half_height
            outlist.append([point.Label, X, Y])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "X (Pixels)", "Y (Pixels)"])
        writer.writerows(outlist)

if __name__ == "__main__":
    nav_file = input("Enter name and path to .nav file: ")
    csv_file = input("Enter name and path to .csv file for save: ")   

    nav_to_csv(nav_file, csv_file)  
        
