from sklearn import svm
import crowdAIparser as ca
import histogram as hist
import numpy as np
import sys
import pickle



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


print "Storing svm..."
with open('hoisvm.pickle','rd') as hoisvm:
   clf = pickle.load(hoisvm)

print "Stored"
test_labels = labels[training_length:]
test_histograms = histograms[training_length:]
true = 0.0
false = 0.0
pred = clf.predict(test_histograms)
for i in range(test_length):
    if pred[i] == test_labels[i]:
        true += 1.0
    else:
        false += 1.0



print "False", false
print "True", true
print "Ratio", true / (true + false)
