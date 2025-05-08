class DataClass:
    def __init__(self):
        self.anoVysyp = -1
        self.maxDuration = float('-inf')
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
        self.listVelocity = []

    def loadRowIntoClass(self, duration, valueX, valueY, valueZ, velocity):
        self.listDuration.append(duration)
        self.listX.append(valueX)
        self.listY.append(valueY)
        self.listZ.append(valueZ)
        self.listVelocity.append(velocity)

    def finishLoading(self, velocityThreshhold):
        self.convertDurationList()

        self.maxX = max(self.listX)
        self.minX = min(self.listX)
        self.maxY = max(self.listY)
        self.minY = min(self.listY)
        self.maxZ = max(self.listZ)
        self.minZ = min(self.listZ)

        self.defineIfVysyp(velocityThreshhold)

    def convertDurationList(self):
        localMinDuration = min(self.listDuration)
        self.maxDuration = max(self.listDuration) - localMinDuration

        for i in range(len(self.listDuration)):
            self.listDuration[i] = self.listDuration[i] - localMinDuration

    def defineIfVysyp(self, velocityThreshhold):
        localAverageVelocity = sum(self.listVelocity) / len(self.listVelocity)
        if localAverageVelocity < velocityThreshhold:
            self.anoVysyp = 1
        else:
            self.anoVysyp = 0