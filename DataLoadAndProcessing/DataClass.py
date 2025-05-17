from datetime import timedelta

class DataClass:
    def __init__(self):
        self.timestamp = None

        self.maxX = float('-inf')
        self.minX = float('inf')
        self.maxY = float('-inf')
        self.minY = float('inf')
        self.maxZ = float('-inf')
        self.minZ = float('inf')

        self.listDuration = []
        self.listX = []
        self.listY = []
        self.listZ = []

    def loadRowIntoClass(self, timestamp, duration, valueX, valueY, valueZ):
        if self.timestamp == None:
            self.timestamp = timestamp

        self.listDuration.append(self.convertDuration(duration))
        self.listX.append(valueX)
        self.listY.append(valueY)
        self.listZ.append(valueZ)

    def finishLoading(self):
        self.maxX = max(self.listX)
        self.minX = min(self.listX)
        self.maxY = max(self.listY)
        self.minY = min(self.listY)
        self.maxZ = max(self.listZ)
        self.minZ = min(self.listZ)

    def convertDuration(self, duration):
        return self.timestamp + timedelta(milliseconds=duration)
