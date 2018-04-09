import glob
import os
import cv2 as cv
import numpy as np
import re

# Sorts file in natural order
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# Timestamp to sample count
def time_to_sample(hour, minute):
    return hour * 60 + minute

def load_data():
    rows, cols = (120, 160) # Adjust for NNs
    files = glob.glob('data_small/*.png')
    files.sort(key=natural_sort_key)

    imgs = np.zeros((1440, rows, cols, 3))
    i = 0

    for file in files:
        img = cv.imread(file)
        imgs[i] = img
        i += 1

    return imgs
