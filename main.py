#import matplotlib.pyplot as plt

#Moje importy
from DataLoadAndProcessing.DataLoadClass import DataLoadClass


if __name__ == '__main__':
    data_load_class = DataLoadClass("D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\2MnouSpracovane\\2_refined.xlsx", "D:\\Skola\\Skola 4 sem ING\\HSU\\Semestralka\\SemestralkaSubory\\3VygenerovaneSubory", 1.0)

    data_load_class.loadData()
    data_load_class.normalizeData()
    data_load_class.exportDataAsImages('Red', 'Green', 'Blue')
