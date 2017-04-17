import cv2
import numpy as np
from matplotlib import pyplot as plt

def getCar(number,csv):
    carImage = cv2.imread('./object-detection-crowdai/'+csv[number][4])
    x = csv[number][0]
    y = csv[number][1]
    h = csv[number][3]-y
    w = csv[number][2]-x
    carImage = carImage[y:y+h,x:x+w]
    return carImage
def getObject(number,csv):
    objectImage = cv2.imread('./object-detection-crowdai/'+csv[number][4])
    objectType = csv[number][5]
    x = csv[number][0]
    y = csv[number][1]
    h = csv[number][3]-y
    w = csv[number][2]-x
    objectImage = objectImage[y:y+h,x:x+w]
    return (objectImage,objectType)

def getImage(number,csv):
    image = cv2.imread('./object-detection-crowdai/'+csv[number][4])
    return image

def loadFile():
    csv = np.recfromcsv('./object-detection-crowdai/labels.csv')
    return csv

