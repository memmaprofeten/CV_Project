from sklearn import svm
import crowdAIparser as ca
import histogram as hist
import numpy as np
import sys
import pickle


def getTrainingVectors(csv):
    length = len(csv)
    hois = np.empty(length)
    labels = np.empty(length)
    for i in range(length):
        if csv[i][4] != csv[i-1][4]: 
            image = ca.getImage(i,csv)
        x = csv[i][0]
        y = csv[i][1]
        h = csv[i][3]-y
        w = csv[i][2]-x
        objectImage = image[y:y+h,x:x+w]
        histogram = hist.hoi(objectImage) 
        hois = np.append(hois,histogram)
        label = ca.getLabel(i,csv)
        if label == 'Car':
            labels[i] = 0
        elif label == 'Truck':
            labels[i] = 1
        elif label == 'Pedestrian':
            labels[i] = 2
        progress(i,length)
    return (hois,labels)
    print "Done with image feature creation"

csv = ca.loadFile()
length = len(csv)
test_length = 1000
training_length = length-test_length
#histograms,labels = getTrainingVectors(csv)

print "Loading data..."
with open('hist.pickle','rd') as hist:
   histograms = pickle.load(hist)

with open('labels.pickle','rd') as label:
    labels = pickle.load(label)

print "Loaded"
clf = svm.SVC(decision_function_shape='ovo')
print "Fitting"
clf.fit(histograms[:training_length],labels[:training_length])
print 'done fitting'

print "Storing svm..."
with open('hoisvm.pickle','wd') as hoisvm:
   histograms = pickle.dump(clf,hoisvm,pickle.HIGHEST_PROTOCOL)

print "Stored"
