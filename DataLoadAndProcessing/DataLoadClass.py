import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import os

from DataLoadAndProcessing.AnotatedDataClass import AnotatedDataClass
from DataLoadAndProcessing.DataClass import DataClass
from DataLoadAndProcessing.ExactDataPointClass import ExactDataPointClass


class DataLoadClass:
    def __init__(self, pathDataFile, pathAnotatedDataFile, pathImageOutputPath):
        self.pathDataFile = pathDataFile
        self.pathAnotatedDataFile = pathAnotatedDataFile
        self.pathImageOutputPath = pathImageOutputPath

        self.listDataClass = []
        self.listAnotatedDataClass = []
        self.listExactDataPointClass = []

        self.globalMaxX = float('-inf')
        self.globalMinX = float('inf')
        self.globalMaxY = float('-inf')
        self.globalMinY = float('inf')
        self.globalMaxZ = float('-inf')
        self.globalMinZ = float('inf')

        self.totalImagesThatCanBeGenerated = 0
        self.totalImagesThatWereGenerated = 0
        self.totalImagesThatWereNotGenerated = 0
        self.totalImagesThatWereGeneratedNegative = 0
        self.totalImagesThatWereGeneratedPositive = 0

    def loadAllData(self):
        print(f'Start')
        print(f'loadAnotatedData')
        self.loadAnotatedData()

        print(f'loadData')
        self.loadData()
        self.normalizeData()

        print(f'createContinuousData')
        self.createContinuousData()
        print(f'Done')

    def loadData(self):
        dataFrame = pd.read_excel(self.pathDataFile, header=None, converters={
            0: pd.to_datetime,
            2: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
            3: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
            4: lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x),
        })

        data_class = DataClass()
        previous_timestamp = None

        for index, row in dataFrame.iterrows():
            current_timestamp = row[0]

            if previous_timestamp is not None and current_timestamp != previous_timestamp:
                data_class.finishLoading()
                self.listDataClass.append(data_class)
                self.updateGlobalVariables(data_class)
                data_class = DataClass()

            data_class.loadRowIntoClass(row[0], row[1], row[2], row[3], row[4])
            previous_timestamp = current_timestamp

        data_class.finishLoading()
        self.listDataClass.append(data_class)
        self.updateGlobalVariables(data_class)

    def loadAnotatedData(self):
        dataFrame = pd.read_excel(self.pathAnotatedDataFile, header=None, converters={
            0: pd.to_datetime,  # vysypStart
            1: pd.to_datetime,  # vysypEnd
        })

        for index, row in dataFrame.iterrows():
            anotatedDataClass = AnotatedDataClass(row[0], row[1])
            self.listAnotatedDataClass.append(anotatedDataClass)

    def updateGlobalVariables(self, data_class):
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

    def createContinuousData(self):
        for data_class in self.listDataClass:
            for i in range(len(data_class.listDuration)):
                exactDataPoint = ExactDataPointClass(data_class.listDuration[i], data_class.listX[i],
                                                     data_class.listY[i], data_class.listZ[i])
                self.listExactDataPointClass.append(exactDataPoint)

        self.listExactDataPointClass.sort(key=lambda x: x.timestamp)

    def exportDataAsImages(self, dataPointsWidth, stride, minPointsForEvent):
        total_points = len(self.listExactDataPointClass)
        num_images = (total_points - dataPointsWidth) // stride + 1
        self.totalImagesThatCanBeGenerated = num_images

        print(f'Generating images')
        for i in range(num_images):
            # print(f'{i} img')

            start_idx = i * stride
            end_idx = start_idx + dataPointsWidth
            chunk = self.listExactDataPointClass[start_idx:end_idx]

            # Prefix a ci sa ma generovat
            prefix, should_generate = self.getImageLabelAndValidity(chunk, minPointsForEvent)
            if not should_generate:
                self.totalImagesThatWereNotGenerated += 1
                continue

            self.totalImagesThatWereGenerated += 1

            X_values = [point.X for point in chunk]
            Y_values = [point.Y for point in chunk]
            Z_values = [point.Z for point in chunk]
            x_axis = range(dataPointsWidth)

            plt.figure(figsize=(5.12, 5.12), dpi=100)
            plt.plot(x_axis, X_values, color='Red')
            plt.plot(x_axis, Y_values, color='Green')
            plt.plot(x_axis, Z_values, color='Blue')

            plt.box(False)
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            plt.xlim(0, dataPointsWidth - 1)
            plt.ylim(0, 1)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            image_path = os.path.join(self.pathImageOutputPath, f'{prefix}img_{i + 1}.png')
            plt.savefig(image_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f'Number of images that could be generated: {self.totalImagesThatCanBeGenerated}')
        print(f'Number of images that were generated: {self.totalImagesThatWereGenerated}')
        print(f'Number of images that were not generated: {self.totalImagesThatWereNotGenerated}')
        print(f'Number of images that were generated negative: {self.totalImagesThatWereGeneratedNegative}')
        print(f'Number of images that were generated positive: {self.totalImagesThatWereGeneratedPositive}')

    def getImageLabelAndValidity(self, chunk, minPointsForEvent):
        timestamps = [point.timestamp for point in chunk]
        start_ts = timestamps[0] #prvy
        end_ts = timestamps[-1] #posledny

        nr_of_events_in_data_window = 0

        for event in self.listAnotatedDataClass:
            # Fast skip: if the chunk is fully before or after the event, skip it
            if end_ts < event.vysypStart or start_ts > event.vysypEnd:
                continue

            # Count how many timestamps are inside the event
            points_in_event = 0
            for t in timestamps:
                if event.vysypStart <= t <= event.vysypEnd:
                    points_in_event += 1
                    if points_in_event >= minPointsForEvent:
                        # print(f'Event datapoints past threshold, events++')
                        break

            if points_in_event >= minPointsForEvent:
                nr_of_events_in_data_window += 1
                if nr_of_events_in_data_window > 1:
                    # print(f'Too many events, skip image')
                    return None, False  # Too many events, skip image

        if nr_of_events_in_data_window == 1:
            # print(f'One event, generating image')
            self.totalImagesThatWereGeneratedPositive += 1
            return "1_", True  # One event detected
        else:
            # print(f'No event, generating image')
            self.totalImagesThatWereGeneratedNegative += 1
            return "0_", True  # No event detected