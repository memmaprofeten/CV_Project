import cv2
import numpy as np
from matplotlib import pyplot as plt
import crowdAIparser as ca

def hoi(image):
    colors = ('b','g','r')
    allHist = np.empty([32*3,1],dtype=int) 
    for i in enumerate(colors):
        hist = cv2.calcHist([car],[i],None,[32],[0,256])
        allhist = np.append(allHist,hist)

    allHistNorm = allHist.astype(float) / float(allHist.max())
    return allHistNorm
