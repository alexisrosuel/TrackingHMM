TrackingHMM
============

![Tracking example](http://i.imgur.com/0E7mwLE.png)


## Idea
Face tracking using HMM and particle filtering based on facial contour and skin color.


## How to use
You can launch the project with the `main.py` script. 


#### Source

With the `--source` argument you can specify the source of the video :
- Pre-loaded video sequence available ine the data folder, for example `--source sequence1`
- Webcam live feed : `--source webcam`

#### Display
If you want to display the full particles and cercles, use the `--fulldisplay` argument.

#### Example
- sequence 1 with full display : `python3 main.py --source sequence1 --fulldisplay`

- live webcam with standard display : `python3 main.py --source webcam`

## Requirements
This scripts runs on Python 3 using the following libraries :
- scikit-image
- opencv
- matplotlib
- numpy
- scipy

## Contributors 
Pascal Jauffret, Alexis Rosuel, Duc-Vinh Tran


