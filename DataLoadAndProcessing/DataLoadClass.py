import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import os

from DataLoadAndProcessing.DataClass import DataClass


class DataLoadClass:
    def __init__(self, pathToLoadFile, pathToSaveFolder, velocityThreshhold):
        self.pathToLoadFile = pathToLoadFile
        self.pathToSaveFolder = pathToSaveFolder
        self.velocityThreshhold = velocityThreshhold

        self.listDataClass = []
        self.globalMaxDuration = float('-inf')
        self.globalMaxX = float('-inf')
        self.globalMinX = float('inf')
        self.globalMaxY = float('-inf')
        self.globalMinY = float('inf')
        self.globalMaxZ = float('-inf')
        self.globalMinZ = float('inf')

    def loadData(self):
        dataFrame = pd.read_excel(self.pathToLoadFile, header=None, converters={
            2: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
            3: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
            4: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
            5: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
        })

        data_class = DataClass()
        previous_timestamp = None

        for index, row in dataFrame.iterrows():
            current_timestamp = row[0]

            if previous_timestamp is not None and current_timestamp != previous_timestamp:
                data_class.finishLoading(self.velocityThreshhold)
                self.listDataClass.append(data_class)
                self.updateGlobalVariables(data_class)
                data_class = DataClass()

            data_class.loadRowIntoClass(row[1], row[2], row[3], row[4], row[5])
            previous_timestamp = current_timestamp

        data_class.finishLoading(self.velocityThreshhold)
        self.listDataClass.append(data_class)
        self.updateGlobalVariables(data_class)

    def updateGlobalVariables(self, data_class):
        if data_class.maxDuration > self.globalMaxDuration:
            self.globalMaxDuration = data_class.maxDuration

        if data_class.maxX > self.globalMaxX:
            self.globalMaxX = data_class.maxX

        if data_class.minX < self.globalMinX:
            self.globalMinX = data_class.minX

        if data_class.maxY > self.globalMaxY:
            self.globalMaxY = data_class.maxY

        if data_class.minY < self.globalMinY:
            self.globalMinY = data_class.minY

        if data_class.maxZ > self.globalMaxZ:
            self.globalMaxZ = data_class.maxZ

        if data_class.minZ < self.globalMinZ:
            self.globalMinZ = data_class.minZ

    def exportDataAsImages(self, colorX, colorY, colorZ):
        for i, data_class in enumerate(self.listDataClass):
            print(f"Image nr.: {i}")
            # if i == 20:
            #     print(f"Image nr. Here: {i}")
            plt.figure(figsize=(5.12, 5.12), dpi=100)

            # Plot the lines for X, Y, Z data
            plt.plot(data_class.listDuration, data_class.listX, color=colorX)
            plt.plot(data_class.listDuration, data_class.listY, color=colorY)
            plt.plot(data_class.listDuration, data_class.listZ, color=colorZ)

            # Disable box around the plot
            plt.box(False)

            # Remove axes labels and ticks if needed
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # X os
            plt.xlim(min(data_class.listDuration), max(data_class.listDuration))
            # Y os
            plt.ylim(min(min(data_class.listX), min(data_class.listY), min(data_class.listZ)), max(max(data_class.listX), max(data_class.listY), max(data_class.listZ)))

            # Remove padding and adjust layout for tight fit
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Fine-tune the margins

            # Save the image
            image_path = os.path.join(self.pathToSaveFolder, f'{data_class.anoVysyp}_img_{i + 1}.png')
            plt.savefig(image_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

    def normalizeData(self):
        for data_class in self.listDataClass:
            # Normalize X
            data_class.listX = [
                (x - self.globalMinX) / (self.globalMaxX - self.globalMinX)
                if self.globalMaxX != self.globalMinX
                else 0.0
                for x in data_class.listX
            ]
            # Normalize Y
            data_class.listY = [
                (y - self.globalMinY) / (self.globalMaxY - self.globalMinY)
                if self.globalMaxY != self.globalMinY
                else 0.0
                for y in data_class.listY
            ]
            # Normalize Z
            data_class.listZ = [
                (z - self.globalMinZ) / (self.globalMaxZ - self.globalMinZ)
                if self.globalMaxZ != self.globalMinZ
                else 0.0
                for z in data_class.listZ
            ]
            # Normalize Duration
            data_class.listDuration = [
                (d - 0.0) / (self.globalMaxDuration - 0.0)
                if self.globalMaxDuration != 0
                else 0
                for d in data_class.listDuration
            ]
