from imutils import face_utils
import numpy as np
#import argparse
import imutils
import dlib
import cv2
import glob
from PIL import Image
#import os

def detect_stroke(video_frame):
    return True

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)


#loading image data sets from folder
def load_image_data(folder):
    imgs = []
    for pic in glob.glob(folder + '/*.jpg'):
        img = Image.open(pic)
        arr = np.array(img)
        imgs.append(arr)
    return imgs


# #takes an array of images from our data sets and applies the dlib model
# #outputs np arr with all facial landmarks of each image
# #these arrays will be the etsting and tarining sets of our model
def extract_facial_landmarks(imgs_array):
    # detect faces in the grayscale image
    facial_landmarks = []
    for img in imgs_array:
        img = imutils.resize(img, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # append the facial landmarks to the list
            facial_landmarks.append(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                
    return np.array(facial_landmarks)


# load the input images
stroke_img = load_image_data('stroke_data')
nonstroke_img = load_image_data('nonstroke_data')

#extracting face landmarks
stroke_facial_landmarks = extract_facial_landmarks(stroke_img)
nonstroke_facial_landmarks = extract_facial_landmarks(nonstroke_img)

print(stroke_facial_landmarks.shape)
print(nonstroke_facial_landmarks.shape)