# Balcony-418
## Introduction
B418 consists of a machine learning experiment, centered around a DIY dataset. A 24-hour timelapse with outside images taken once a minute comprises the data collection. On it, a number of experiments related to data analysis and machine learning will be performed. The _b418.py_ module contains the functionality needed to easily operate the fresh dataset. The machine learning experiments will only make sense when using similar images.

## Build your dataset
To create your own dataset, connect your camera (which must support live USB feed) to your computer. Find the identifier of your camera. Modify the _capture.py_ file accordingly, by adding the identifier at line 4. Run the script:
```
python capture.py
```
You should see a new window and the live camera feed. If you see a black window, try testing the camera in an external application first. In the console, you will be prompted with a message whenever a frame is captured. Sit back and relax. Once completed, you should downscale your dataset images into a smaller version for faster NN experiments.

If you can't create your own dataset, try this one (TODO). For the NN experiments, the images have been downscaled to a heart-breaking 160x120 resolution. 
![sample.png](https://github.com/paubric/Balcony-418/blob/master/sample.png)

## Experiments
### Light Intensity Analysis
Computes the average RGB value of every pixel for every image and plots the resulted series, together with a smoothed version. 
```
python brightness.py
```
Although not measuring the actual light intensity at every wavelength, there is a strong correlation between that and the RGB value average. A high plateau can be noticed at daytime and a lower one at nighttime. The sunrise and sunset can be observed easily through the high absolute value of the slope. Lines representing the ground-truth sunrise and sunset timestamps have been added in order to validate the observation.
![Figure_time_prediction.png](https://github.com/paubric/Balcony-418/blob/master/Figure_brightness.png)

### Average Image
Computes the average RGB value for every pixel in every image and saves the resulted image.
```
python average_image.py
```
At a first glance, the picture has a moderate brightness, which is due to the fairly balanced day-night ratio. An interesting observation is the existence of several ghost objects (see the red rectangles). The longer the time in which an object has been present in the picture, the stronger it is defined.

![average_image_rect.png](https://github.com/paubric/Balcony-418/blob/master/average_image_rect.png)

### Predict Timestamp
Trains a Tensorflow model, through Keras, to predict the moment of the day at which an image has been captured.
```
python predict_time.py
```
The model consists of several 2D convolutional layers (with 3D filters for color), followed by pooling, dropout and flatten layers. The final neuron will output a number between 0 and 1, which is later scaled to the 0-24h interval. During daytime, the model performs very well by learning the changing patterns (probably being heavily influenced by the building shadows). The model has some difficulty in making precise predictions during nighttime because of the low number of features which discriminate the night images.  
![Figure_time_prediction.png](https://github.com/paubric/Balcony-418/blob/master/Figure_time_prediction.png)

### Predict Temperature
Trains a Tensorflow model, through Keras, to predict the air temperature at the moment when an image has been captured.
```
python predict_temperature.py
```
The model consists of several 2D convolutional layers (with 3D filters for color), followed by pooling, dropout and flatten layers. The final neuron will output a number between 0 and 1, which is later denormalized to a temperature value. The model closely follows the temperature evolution.
![Figure_temperature_prediction.png](https://github.com/paubric/Balcony-418/blob/master/Figure_temperature_prediction.png)

### Time Walk
Given an input image, shift it by ±6/12h.

### Car Detection
Plot number of cars versus time, observe hours with most parked cars.

## TODO
- Link demo dataset
- Split the dataset into a training set and a testing set (50% and 50%), alternatively distribute images
- Implement Time Walk experiment
- Implement Car Detection experiment
- Optionally downscale automatically for NNs
- Remove hardcoded temperature data
