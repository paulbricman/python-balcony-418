# Balcony-418
## Introduction
B418 consists of a machine learning experiment, centered around a DIY dataset. A 24-hour timelapse with images taken once a minute comprises the data collection. 

## Build your dataset
Connect your camera which supports live USB feed to your computer. Find the identifier of your camera. Modify the _capture.py_ file accordingly. Run the script:
```
python capture.py
```
You should see a new window and the live camera feed. In the console, you will be prompted when a frame is captured. Grab a coffee. 

If you don't have the possibility of creating your own dataset, try this one (TODO).

## Experiments
### Light Intensity Analysis
Computes the average RGB value of every pixel for every image and plots the resulted series, together with a smoothed version. Although not measuring the actual light intensity at every wavelength, there is a strong correlation between that and the RGB value average. A high plateau can be noticed at daytime and a lower one at nighttime. The sunrise and sunset can be observed easily through the high absolute value of the slope. Lines representing the ground-truth sunrise and sunset timestamps have been added in order to validate the observation.
![Figure_brightness.png](https://github.com/paubric/Balcony-418/blob/master/Figure_brightness.png)

### Average Image
Computes the average RGB value for every pixel in every image and saves the resulted image. From a first glance, the picture has a moderate brightness, which is due to the fairly balanced day-time ratio. An interesting observation is the existence of several ghost objects (see the red rectangles). The longer the time in which an object has been present in the picture, the stronger it is defined.
![average_image_rect.png](https://github.com/paubric/Balcony-418/blob/master/average_image_rect.png)

### Predict timestamp
Trains a Tensorflow model, through Keras, to predict the moment of the day at which an image has been captured. The model consists of several 2D convolutional layers (with 3D filters for color), followed by pooling, dropout and flatten layers. The final neuron will output a number between 0 and 1, which is later scaled to the 0-24h interval. During daytime, the model performs very well by learning the changing patterns (probably being heavily influenced by the building shadows). The model has some difficulty in making precise predictions during nighttime because of the low number of features which discriminate the night images.  
![Figure_time_prediction.png](https://github.com/paubric/Balcony-418/blob/master/Figure_time_prediction.png)

### Artificial Synesthesia
Given an input image and temperature data analogous to the image dataset, predict the air temperature. 

### Time Walk
Given an input image, shift it by ±6/12h.
