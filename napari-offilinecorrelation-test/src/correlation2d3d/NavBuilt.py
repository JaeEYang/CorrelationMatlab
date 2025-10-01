## Structured data model for building a new Nav file

import numpy as np

class MapItem:
    def __init__(self):
        self.Label = ""
        self.Color = 0
        self.StageXYZ = [0.0, 0.0, 0.0]
        self.NumPts = 0
        self.Regis = 1
        self.Type = 2

        self.MapFile = ""
        self.MapID = 0
        self.MapMontage = 0
        self.MapSection = 0
        self.MapBinning = 0
        self.MapMagInd = 0
        self.MapCamera = 0

        # Transform & geometry
        self.MapScaleMat = np.eye(2)   # 2x2 matrix
        self.MapWidthHeight = [2048, 2048] # default F4 camera
        self.RawStageXY = [0.0, 0.0]

        # Outline coordinates
        self.PtsX = []
        self.PtsY = []

        # Linked points
        self.Points = []

    def to_dict(self):
        return {
            "Label": self.Label,
            "MapID": self.MapID,
            "StageXYZ": self.StageXYZ,
            "MapFile": self.MapFile,
            "MapWidthHeight": self.MapWidthHeight,
        }


class PointItem:
    def __init__(self):
        self.Label = ""
        self.Color = 0
        self.StageXYZ = [0.0, 0.0, 0.0]
        self.NumPts = 1
        self.Regis = 1
        self.Type = 0
        self.GroupID = 0
        self.Imported = 0
        self.OrigReg = 0
        self.DrawnID = 0
        self.PtsX = 0.0
        self.PtsY = 0.0

    def getText(self):
        text = [f"[Item = {self.Label}]"]
        text.append(f"Color = {self.Color}")
        text.append("StageXYZ = {:.3f} {:.3f} {:.3f}".format(*self.StageXYZ))
        text.append(f"NumPts = {self.NumPts}")
        text.append(f"Regis = {self.Regis}")
        text.append(f"Type = {self.Type}")
        text.append(f"GroupID = {self.GroupID}")
        text.append(f"Imported = {self.Imported}")
        text.append(f"OrigReg = {self.OrigReg}")
        text.append(f"DrawnID = {self.DrawnID}")
        text.append("PtsX = {:.3f}".format(self.PtsX))
        text.append("PtsY = {:.3f}".format(self.PtsY))
        return text

    def to_dict(self):
        return {
            "Label": self.Label,
            "StageXYZ": self.StageXYZ,
            "PtsX": self.PtsX,
            "PtsY": self.PtsY,
            "DrawnID": self.DrawnID,
        }


class NavData:
    def __init__(self):
        self.Maps = []
        self.Points = []
        self.MapDictionary = {}

    def addMapItem(self, mapItem: MapItem):
        self.Maps.append(mapItem)
        self.MapDictionary[mapItem.MapID] = mapItem

    def addPointItem(self, pointItem: PointItem):
        self.Points.append(pointItem)
        if pointItem.DrawnID in self.MapDictionary:
            self.MapDictionary[pointItem.DrawnID].Points.append(pointItem)

    def writeHeader(self, file, filename):
        file.write("AdocVersion = 2.00\n")
        file.write("LastSavedAs = " + filename + "\n\n")

    def writeItems(self, file):
        """TODO: Write each item correctly back to nav file format."""
        raise NotImplementedError
    
def createRecordInNavData(navdata: NavData, item: dict):
    t = item.get("Type")
    if t == "2":  # Map
        m = MapItem()
        m.Label = item.get("Index", "")
        m.Color = int(item.get("Color", 0))
        if "StageXYZ" in item:
            m.StageXYZ = [float(x) for x in item["StageXYZ"].split()]
        m.NumPts = int(item.get("NumPts", 0))
        m.Regis = int(item.get("Regis", 1))
        m.MapID = int(item.get("MapID", 0))

        if "MapFile" in item:
            m.MapFile = item["MapFile"]
        if "RawStageXY" in item:
            m.RawStageXY = [float(x) for x in item["RawStageXY"].split()]
        if "MapScaleMat" in item:
            nums = [float(x) for x in item["MapScaleMat"].split()]
            m.MapScaleMat = np.array(nums).reshape(2, 2)
        if "MapWidthHeight" in item:
            m.MapWidthHeight = [float(x) for x in item["MapWidthHeight"].split()]
        if "PtsX" in item:
            m.PtsX = [float(x) for x in item["PtsX"].split()]
        if "PtsY" in item:
            m.PtsY = [float(x) for x in item["PtsY"].split()]
        navdata.addMapItem(m)

    elif t == "0":  # Point
        p = PointItem()
        p.Label = item.get("Index", "")
        p.Color = int(item.get("Color", 0))
        if "StageXYZ" in item:
            p.StageXYZ = [float(x) for x in item["StageXYZ"].split()]
        p.NumPts = int(item.get("NumPts", 1))
        p.Regis = int(item.get("Regis", 1))
        p.GroupID = int(item.get("GroupID", 0))
        p.Imported = int(item.get("Imported", 0))
        p.OrigReg = int(item.get("OrigReg", 0))
        if "PtsX" in item:
            p.PtsX = float(item["PtsX"])
        if "PtsY" in item:
            p.PtsY = float(item["PtsY"])
        p.DrawnID = int(item.get("DrawnID", 0))
        navdata.addPointItem(p)
    else:
        print("Unhandled item type:", t)

def parseNavFile(filename: str) -> NavData:
    rv = NavData()
    currentItem = {}
    parsingItem = False

    with open(filename, "r") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                if currentItem:
                    createRecordInNavData(rv, currentItem)
                    currentItem = {}
                continue
            if line.startswith("[Item "):
                if currentItem:
                    createRecordInNavData(rv, currentItem)
                    currentItem = {}
                index = line.split("[Item = ")[1].split("]")[0]
                currentItem["Index"] = index
                parsingItem = True
            elif parsingItem:
                if "=" in line:
                    key, value = [s.strip() for s in line.split("=", 1)]
                    currentItem[key] = value

    if currentItem:
        createRecordInNavData(rv, currentItem)
    return rv

def invert_matrix(mat):
    return np.linalg.inv(np.array(mat, dtype=float))

def multiply_matrix(mat1, mat2):
    return np.dot(np.array(mat1, dtype=float), np.array(mat2, dtype=float))