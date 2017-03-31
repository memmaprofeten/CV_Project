import cv2
import numpy as np
from matplotlib import pyplot as plt

##Simple implementation of template matching
##Just change templatenumber to find different templates
##And same for image
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
    
def loadUdacityFile(filename):
    csv = np.recfromcsv(filename)
    return csv
#plt.imshow(image)
#plt.show()

imageNumber = 0
templateNumber = 100 

csv = loadUdacityFile("./object-detection-crowdai/labels.csv")
image = getImage(imageNumber,csv)
template,_ = getObject(templateNumber,csv)
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
template = cv2.resize(template, (0,0), fx=0.5, fy=0.5) 
 
# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  
# Find template
result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
print max_val
h,w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)
   
# Show result
cv2.imshow("Template", template)
cv2.imshow("Result", image)

cv2.moveWindow("Template", 10, 50);
cv2.moveWindow("Result", 150, 50);
 
cv2.waitKey(0)
