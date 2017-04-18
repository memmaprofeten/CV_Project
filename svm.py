from sklearn import svm
import crowdAIparser as ca
import histogram as hist
import numpy as np
import sys
import pickle

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


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
        histogram = np.asarray(hist.hoi(objectImage)) 
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
histograms,labels = getTrainingVectors(csv)

histograms = histograms.reshape((len(csv),97))

print histograms.shape
print labels.shape
print "Storing data..."
with open('hist.pickle','wb') as hist:
    pickle.dump(histograms,hist,pickle.HIGHEST_PROTOCOL)

with open('labels.pickle','wb') as label:
    pickle.dump(labels,label,pickle.HIGHEST_PROTOCOL)

print "Stored"
