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
### Predict timestamp
Given an input image, predict the time when it was taken.

### Artificial Synesthesia
Given an input image and temperature data analogous to the image dataset, predict the air temperature.

### Time Walk
Given an input image, shift it by ±6/12h.
