import pickle
import sys


filename = sys.argv[1]

with open('hogsvm.pickle','rd') as f:
    clf = pickle.load(f)
