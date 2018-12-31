from data_collector import getDataFromRemote, splitData
from adnormality_detection import findAbnormalities, getGKD

def mainApp():
    jsonData = getDataFromRemote()
    tempData, attributes, encodedValues = splitData(jsonData)
    findAbnormalities(tempData)
    getGKD(tempData)
    return 1