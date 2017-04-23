import cv2
import numpy as np
from matplotlib import pyplot as plt
import crowdAIparser as ca
import math

def hoi(image):
    colors = ('b','g','r')
    allHist = np.empty([32*3,1],dtype=int) 
    for i in range(2):
        hist = cv2.calcHist([image],[i],None,[32],[0,256])
        allhist = np.append(allHist,hist)

    allHistNorm = allHist.astype(float) / float(allHist.max())
    return allHistNorm

def hog(image):
    shape = image.shape
    #If image is not sqare crop 
    #according to smaller dimension
    if shape[0] != shape[1]:
        if shape[0] > shape[1]:
            image = image[0:shape[1],0:shape[1]]
        else:
            image = image[0:shape[0],0:shape[0]]
    image = np.float32(image) / 255.0
    #Find gradients for all pixels
    gx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=1)
    gy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #Simple Histogram of gradients
    angle = np.float32(angle) % 180.0 #Unsigned gradients
    histograms = np.zeros((9,3))
    hej = 0
    for i in range(mag.shape[2]):
        for j in range(1,mag.shape[1]-1):
            for k in range(1,mag.shape[0]-1):
                section = int(math.floor(angle[k,j,i]/20))
                amount = (angle[k,j,i] % 20)/20.0
                histograms[section,i] += mag[k,j,i] * (1.0-amount)
                if amount < 0.5:
                    section -= 1
                else:
                    section += 1
                    if section == 9:
                        section = 0
                if amount != 0:
                    histograms[section,i] += mag[k,j,i] * (amount)
    histograms = histograms.flatten()
    histograms = histograms / np.amax(histograms) 
    return histograms
