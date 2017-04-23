import pickle
import sys
import histogram as hs
import cv2
filename = sys.argv[1]

with open('hogsvm.pickle','rb') as f:
    clf = pickle.load(f)

image = cv2.imread(filename)
print("Looking")
size = 40
x,y = size,size
while x+size < image.shape[0] and y+size < image.shape[1]/2:
    test = image[y:y+size,x:x+size]
    hog = hs.hog(test)
    clf.predict(hog)
    if pred == 1:
        print("Found something")
        
    y += 10
    x += 10

