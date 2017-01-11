TrackingHMM
============

## Idea
Face tracking using HMM and particle filtering based on facial contour and skin color.


## How to use
You can launch the project with the `main.py` script. 


### Source

With the `--source` argument you can specify the source of the video :
- Pre-loaded video sequence available ine the data folder, for example `--source sequence1`
- Webcam live feed : `--source webcam`

### Display
If you want to display the full particles and cercles, use the `--fulldisplay` argument.

## Requirements
This scripts runs on Python 3 using the following libraries :
- scikit-image
- opencv
- matplotlib
- numpy
- scipy

