import histogram as hs
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import pickle
from sklearn import svm
import glob
import pandas as pd
from skimage import io, transform, exposure
import sys
import random
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def process_img(img):
    img = transform.resize(img, (img_rows, img_cols), mode='constant')
    img = contrast_stretch(img)
    return img

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels
def contrast_stretch(img):
    p2, p98 = np.percentile(img, (2, 98))
    for k in range(3):
        img[:,:,k] = exposure.rescale_intensity(img[:,:,k], in_range=(p2, p98))
    return img

img_rows, img_cols = 32, 32
nb_classes = 43
print("Getting images and labels")
images, labels = readTrafficSigns('GTSRB-2/Final_Training/Images')

for label in labels:
    label = 0

print("Getting training images for no sign")
fullimages = open('./TrainIJCNN2013/gt.csv')
gtReader = csv.reader(fullimages, delimiter=';')
random.seed()
i = 0
for row in gtReader:
    background = cv2.imread('./TrainIJCNN2013/'+row[0])
    shape = background.shape
    size = 30
    padding = size + 10
    y = random.randint(0,shape[1]-padding)
    x = random.randint(0,shape[0]-padding)
    background = background[y:y+size,x:x+size]
    images.append(background)
    labels.append(1)
    i += 1

print("Create HOGs for images")
length = len(images)
for i in range(length):
    images[i] = hs.hog(images[i])
    progress(i,length)


with open('hist.pickle','wb') as hist:
    pickle.dump(images,hist,pickle.HIGHEST_PROTOCOL)

with open('label.pickle','wb') as label:
    pickle.dump(labels,label,pickle.HIGHEST_PROTOCOL)

print("Loading data...")
with open('hist.pickle','rb') as hist:
   histograms = pickle.load(hist)

with open('label.pickle','rb') as label:
    labels = pickle.load(label)

print("Loaded")

clf = svm.SVC()
print("Fitting")
clf.fit(histograms,labels)
print('done fitting')


print("Storing svm...")
with open('hogsvm.pickle','wb') as hogsvm:
   pickle.dump(clf,hogsvm,pickle.HIGHEST_PROTOCOL)
print("Stored")

print("Getting test images")
test_dir = 'GTSRB/Final_Test/Images/'
test_images_paths = glob.glob(test_dir + '*.ppm', recursive=True)
test_images = []
test_csv = pd.read_csv('GT-final_test.csv',sep=';')
test_labels = list(test_csv['ClassId'])

for filename in test_images_paths:
    try:
        img = process_img(io.imread(filename))
        test_images.append(img)
    except (IOError, OSError):
        print('missed', filename)
        pass
                                          
X_test = np.array(test_images, dtype='float32')
Y_test = np.eye(nb_classes, dtype='uint8')[test_labels] 
print("Creating Hog for test")
testHogs = np.empty((len(X_test),27))
for i in range(len(X_test)):
    testHogs[i] = hs.hog(X_test[i])


print("Storing test hog...")
with open('hogtest.pickle','wb') as hogtest:
   pickle.dump(testHogs,hogtest,pickle.HIGHEST_PROTOCOL)
print("Stored")

print("Storing test labesls...")
with open('testlabel.pickle','wb') as testlabel:
   pickle.dump(Y_test,testlabel,pickle.HIGHEST_PROTOCOL)
print("Stored")

print("Loading svm..")
with open('hogsvm.pickle','rb') as hogsvm:
    clf = pickle.load(hogsvm)
print("Loaded")

print("Loading test hog...")
with open('hogtest.pickle','wb') as hogtest:
   X_test = pickle.load(hogtest)
print("Loaded")

print("Loading test labesls...")
with open('testlabel.pickle','wb') as testlabel:
   Ytest = pickle.load(testlabel)
print("Loaded")

true = 0
false = 0
pred = clf.predict(X_test)
for i in range(len(X_test)):
    if pred[i] == Y_test[i]:
        true += 1
    else:
        false += 1

print("False", false)
print("True", true)
print("Ratio", true / (true + false))

