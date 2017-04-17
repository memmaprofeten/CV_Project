from sklearn import svm
import crowdAIparser as ca
import histogram as hist
import numpy as np
import sys
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
        image,label = ca.getObject(i,csv)
        histogram = hist.hoi(image) 
        hois = np.append(hois,histogram)
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

clf = svm.SVC(decision_function_shape='ovo')
print "Fitting"
clf.fit(histograms[:training_length],labels[:training_length])
print 'done fitting'

test_labels = labels[training_length:]
test_histograms = histograms[training_length]
true = 0
false = 0
for i in range(test_length):
    pred = clf.predict(test_historgrams[i])
    if pred == test_labels[i]:
        true += 1
    else:
        false += 1



print "False", false
print "True", true
print "Ratio", true / (true + false)
