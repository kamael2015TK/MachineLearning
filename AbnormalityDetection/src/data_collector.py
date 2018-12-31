import urllib.request
import json
import numpy as np
from scipy import stats

def getIndexFromList(targetList: [], targetObject=''):
    if targetObject in targetList: 
        return targetList.index(targetObject)
    else:
        return -1

def helper(propertyName = '', dataObject = {}):
    return dataObject.get(propertyName) or ''

def getTargetProperties(): 
    return ['countryCode', 'agentCode', 'timestamp', 'value']

def encodeCountryCode(encodedList = [], value = ''):
    if value == '':
        return encodedList, -1
    index = -1
    if value in encodedList:
        index = encodedList.index(value)
    else:
        index = len(encodedList)
        encodedList.append(value)
    return encodedList, ( index + 1 )


def getDataFromRemote():
    data = urllib.request.urlopen('https://api.myjson.com/bins/hgka4').read()
    return json.loads(data)

def standardizeData(tempData=[], rangeOfNormalAttributes = []): 
    tempData[:,0] = pow(2, tempData[:,0])
    tempData[:,1] = pow(2, tempData[:,1])
    tempData[:,rangeOfNormalAttributes] = stats.zscore(tempData[:,rangeOfNormalAttributes], ddof=0)
    return tempData

# takes the data fetches predifined attributes and l
def splitData(data):
    currency = data[0]['currency']
    attributes = getTargetProperties()
    tempData = np.empty((len(data), len(attributes)))
    encoded = []
    for i, entry in enumerate(data):
        for j, attribute in enumerate(attributes):
            if attribute == 'countryCode':
                encoded, tempData[i,j] = encodeCountryCode(encoded, helper(attribute, entry))
            else: 
                tempData[i,j] = helper(attribute, entry)
    zscoreRange = range(2,len(attributes))
    tempData = standardizeData(tempData, zscoreRange)
    return tempData, attributes, encoded






