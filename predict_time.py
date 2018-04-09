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

# Input image dimensions
img_rows, img_cols = 120, 160
input_shape = (img_rows, img_cols, 3)

# Hyperparameters
batch_size = 60
epochs = 12

# Data
x_train = b418.load_data()
y_train = np.array(())

# Sample count normalization
for i in range(1440):
    y_train = np.append(y_train, [i / 1440])

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
if os.path.exists("predict_time.h5") == False:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True)

    model.save_weights('predict_time.h5')
else:
    # Load load_weights
    model.load_weights('predict_time.h5')

# Predict, denormalize, smoothing
predictions = np.multiply(model.predict(x_train), 24)
smooth_predictions = gaussian_filter(predictions, 30)

# Plot the series
plt.plot(predictions, 'b')
plt.plot(smooth_predictions, 'c')
plt.plot([0, 1440], [0, 24], 'y--')

plt.title('Time Prediction (Train)')
plt.legend(('original', 'smooth', 'ideal'))
plt.show()
