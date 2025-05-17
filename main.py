#import matplotlib.pyplot as plt

#Moje importy
from DataLoadAndProcessing.DataLoadClass import DataLoadClass


if __name__ == '__main__':
    pathDataFile30 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_refined_30.xlsx"
    pathDataFile35 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_refined_35.xlsx"
    pathAnotatedDataFile30 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_AnotatedData_refined_30.xlsx"
    pathAnotatedDataFile35 = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\5SpracovaneDataNovyPostup\\2_AnotatedData_refined_35.xlsx"
    pathImageOutputPath = "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\3VygenerovaneSubory"
    dataPointsWidth = 60
    stride = 20
    minPointsForEvent = 10

    data_load_class = DataLoadClass(pathDataFile30, pathAnotatedDataFile30, pathImageOutputPath)
    data_load_class.loadAllData()
    data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)

    # data_load_class = DataLoadClass(pathDataFile35, pathAnotatedDataFile35, pathImageOutputPath)
    # data_load_class.loadAllData()
    # data_load_class.exportDataAsImages(dataPointsWidth, stride, minPointsForEvent)


# -------------------------------------
# 50
# 25
# 10
# 1100 negativnych 100 pozitivnych

#---------------------------------------
# 20
# 5
# 10
# 5788 negativnych 203 pozitivnych

#---------------------------------------
# 100
# 20
# 20
# 1447 negativnych 47 pozitivnych

#---------------------------------------
# 250
# 75
# 20
# 369 negativnych 24 pozitivnych 4 nevygenerovane

#---------------------------------------
# 200
# 100
# 10
# 236 negativnych 35 pozitivnych 27 nevygenerovane

#---------------------------------------
# 60
# 20
# 10
# 1349 negativnych 145 pozitivnych 2 nevygenerovane