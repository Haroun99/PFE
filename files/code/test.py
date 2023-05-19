from config import fcn_config as cfg
from config import fcn8_cnn as cnn
import tensorflow as tf
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time, warnings
from datetime import datetime #DB
import pandas as pd
from tensorflow.keras.backend               import set_session
from tensorflow.keras                       import backend #, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils                 import plot_model #DB
'''HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
EPOCHS = cfg.EPOCHS
UPSCALE=False

dir_data = cfg.DATASET_DIR
dir_train_img = cfg.dir_train_img
dir_train_seg = cfg.dir_train_seg
dir_test_img  = cfg.dir_test_img
dir_test_seg  = cfg.dir_test_seg
dir_calib_img = cfg.dir_calib_img
dir_calib_seg = cfg.dir_calib_seg
dir_valid_img = cfg.dir_valid_img
dir_valid_seg = cfg.dir_valid_seg
model = cnn.FCN8(nClasses     = N_CLASSES,
             input_height = HEIGHT,
             input_width  = WIDTH,
                 upscale = UPSCALE)
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations  = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid = []
Y_valid = []

for im , seg in zip(valid_images,valid_segmentations) :
    X_valid.append( cnn.NormalizeImageArr(os.path.join(dir_valid_img, im)) )
    Y_valid.append( cnn.LoadSegmentationArr( os.path.join(dir_valid_seg, seg), N_CLASSES , WIDTH, HEIGHT)  )
X_valid, Y_valid = np.array(X_valid) , np.array(Y_valid)
print("\n")
print("validation data (X) (Y) shapes:", X_valid.shape,Y_valid.shape)
test_images = os.listdir(dir_test_img)
test_images.sort()
test_segmentations  = os.listdir(dir_test_seg)
test_segmentations.sort()
X_test = []
Y_test = []
for im , seg in zip(test_images,test_segmentations) :
    X_test.append( cnn.NormalizeImageArr(os.path.join(dir_test_img,im)) )
    Y_test.append( cnn.LoadSegmentationArr(os.path.join(dir_test_seg, seg), N_CLASSES , WIDTH, HEIGHT)  )
X_test, Y_test = np.array(X_test) , np.array(Y_test)
print("testing    data (X) (Y) shapes", X_test.shape,Y_test.shape)
print("\n")

if UPSCALE==False :
    model_filename= "../keras_model/fcn8/ep"    + str(EPOCHS) + "_trained_fcn8_"    + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"
else :
    model_filename= "../keras_model/fcn8ups/ep" + str(EPOCHS) + "_trained_fcn8ups_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"'''

#model = load_model(model_filename) #DB

video_path='office.MOV'

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error opening video file")
    exit()
# Read and display frames from the video
while True:
# Read the next frame
    ret, frame = video.read()
    if ret is None:
        print("Error reading")
        exit()
# Display the frame
    cv2.imshow("Video Frame", frame)

# Wait for the 'q' key to be pressed to exit
    if cv2.waitKey(1) == ord('q'):
        break

