# Reference: http://dlib.net/face_landmark_detection.py.html

# run this in repo root directory: python scripts/train_model_cnn.py

# 1. Load dataset and dlib pre-trained model
# 2. Image pre-process: detect, crop and align faces
# 3. Model training: basic CNN
# 4. Save model: save model in directory for prediction

import os
import glob
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 5 face landmarks model is used to crop and align faces
predictor_fl5_path = "model/shape_predictor_5_face_landmarks.dat"

stroke_folder_path = "dataset/stroke"
non_stroke_folder_path = "dataset/non_stroke"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
fl5_detector = dlib.get_frontal_face_detector()
fl5_sp = dlib.shape_predictor(predictor_fl5_path)

non_stroke_labeled_paths = [(f, False) for f in glob.glob(os.path.join(non_stroke_folder_path, "*.jpg"))]
stroke_labeled_paths = [(f, True) for f in glob.glob(os.path.join(stroke_folder_path, "*.jpg"))]

labeled_paths = stroke_labeled_paths + non_stroke_labeled_paths

non_stroke_images = []
stroke_images = []

# extract features for all dataset
for f, is_stroke in labeled_paths:
    # Load the image using Dlib
    img = dlib.load_rgb_image(f)
    
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    fl5_dets = fl5_detector(img, 1)

    num_faces = len(fl5_dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(f))
        continue

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in fl5_dets:
        faces.append(fl5_sp(img, detection))

    # Get aligned and cropped face
    aligned_img = dlib.get_face_chip(img, faces[0])

    if is_stroke == 1:
        stroke_images.append(aligned_img)
    else:
        non_stroke_images.append(aligned_img)

stroke_images = np.array(stroke_images)
non_stroke_images = np.array(non_stroke_images)
print("Finish feature extration")

print("Number of stroke data:", len(stroke_images))
print("Number of non stroke data:", len(non_stroke_images))

# 1 = stroke 0 = non_stroke
stroke_labels = np.array([1] * len(stroke_images))
# stroke_dataset = tf.data.Dataset.from_tensor_slices((stroke_features, stroke_labels))

non_stroke_labels = np.array([0] * len(non_stroke_images))
# non_stroke_dataset = tf.data.Dataset.from_tensor_slices((non_stroke_features, non_stroke_labels))

all_images = np.concatenate((stroke_images, non_stroke_images))
all_labels = np.concatenate((stroke_labels, non_stroke_labels))

import random
zipped_images_labels = list(zip(all_images, all_labels))
random.shuffle(zipped_images_labels)
all_images, all_labels = zip(*(zipped_images_labels))

all_images = [img / 255.0 for img in all_images]

all_images = np.array(all_images)
all_labels = np.array(all_labels)

test_valid_number = 25
test_images = all_images[0:test_valid_number]
test_labels = all_labels[0:test_valid_number]

validate_images = all_images[test_valid_number:test_valid_number + test_valid_number]
validate_labels = all_labels[test_valid_number:test_valid_number + test_valid_number]

train_images = all_images[test_valid_number + test_valid_number:]
train_labels = all_labels[test_valid_number + test_valid_number:]

print(len(test_images))
print(len(validate_images))
print(len(train_images))

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(16)
validate_dataset = tf.data.Dataset.from_tensor_slices((validate_images, validate_labels)).batch(16)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(16)


from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=all_images[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer and fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Optional: Add dropout for regularization
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=validate_dataset, batch_size=16)
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

model.save('model/cnn.keras')