from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import b418
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from itertools import repeat

def normalize(list):
    min_val = min(list)
    max_val = max(list)
    diff = max_val - min_val
    list = [x-min_val for x in list]
    list = [x/diff for x in list]
    return np.array(list)

def denormalize(list, original):
    min_val = min(original)
    max_val = max(original)
    diff = max_val - min_val
    list = [x*diff for x in list]
    list = [x+min_val for x in list]
    return np.array(list)

# Input image dimensions
img_rows, img_cols = 120, 160
input_shape = (img_rows, img_cols, 3)

# Hyperparameters
batch_size = 60
epochs = 12

# Data
x_train = b418.load_data()

# Hardcoded temperature data (TODO)
y_train = [12, 11, 11, 9, 9, 8, 8, 8, 9, 12, 13, 15, 16, 17, 18, 19, 19, 19, 19, 18, 16, 14, 13, 12]

# Preprocessing, temperature normalization
y_train = [x for item in y_train for x in repeat(item, 60)]
print(x_train[720])
print(normalize(y_train))

# RGB normalization
x_train = x_train.astype('float32')
x_train /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# Build model
model = Sequential()
model.add(Conv2D(64, (5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Only fit if model is not saved
if os.path.exists("predict_temperature.h5") == False:
    model.fit(x_train, normalize(y_train),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True)

    model.save_weights('predict_temperature.h5')
else:
    model.load_weights('predict_temperature.h5')

# Predict, denormalize, smoothing
predictions = denormalize(model.predict(x_train), y_train)
smooth_predictions = gaussian_filter(predictions, 30)

# Plot the series
plt.plot(predictions, 'b')
plt.plot(smooth_predictions, 'c')
plt.plot(y_train, 'y--')

plt.title('Temperature Prediction (Train)')
plt.legend(('original', 'smooth', 'ideal'))
plt.show()
