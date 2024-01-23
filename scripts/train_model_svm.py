# Reference: http://dlib.net/face_landmark_detection.py.html

# run this in repo root directory: python scripts/train_model_svm.py

# 1. Load dataset and dlib pre-trained model
# 2. Image pre-process: detect, crop and align faces, generate 68 facial landmarks
# 3. Feature extration: generate 5 features from landmarks, eyes, nose, mouth, eyebrows asymmetry etc.
# 4. Model training: simple SVM
# 5. Model evaluation: k-fold Cross-Validation + cofusion matrix (plot graph here)
# 6. Save model: save model in directory for prediction

import os
import glob
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

import sys
sys.path.append('src/Prediction')
from face_feature import Feature # not proper way

# 5 face landmarks model is used to crop and align faces
# 68 face landamarks model is used to detect precise landmark points on faces
predictor_fl5_path = "model/shape_predictor_5_face_landmarks.dat"
predictor_fl68_path = "model/shape_predictor_68_face_landmarks.dat"

stroke_folder_path = "dataset/stroke"
non_stroke_folder_path = "dataset/non_stroke"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
fl5_detector = dlib.get_frontal_face_detector()
fl5_sp = dlib.shape_predictor(predictor_fl5_path)

fl68_detector = dlib.get_frontal_face_detector()
fl68_sp = dlib.shape_predictor(predictor_fl68_path)

non_stroke_labeled_paths = [(f, False) for f in glob.glob(os.path.join(non_stroke_folder_path, "*.jpg"))]
stroke_labeled_paths = [(f, True) for f in glob.glob(os.path.join(stroke_folder_path, "*.jpg"))]

labeled_paths = stroke_labeled_paths + non_stroke_labeled_paths

non_stroke_features = []
stroke_features = []

def pose_estimate(image, landmarks):
    """
    Given an image and a set of facial landmarks generates the direction of pose
    """
    size = image.shape
    image_points = np.array([
        (landmarks[33].x, landmarks[33].y),     # Nose tip
        (landmarks[8].x, landmarks[8].y),       # Chin
        (landmarks[36].x, landmarks[36].y),     # Left eye left corner
        (landmarks[45].x, landmarks[45].y),     # Right eye right corner
        (landmarks[48].x, landmarks[48].y),     # Left Mouth corner
        (landmarks[54].x, landmarks[54].y)      # Right mouth corner
        ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)     
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    return angles

# extract features for all dataset
for f, is_stroke in labeled_paths:
    # Load the image using Dlib
    img = dlib.load_rgb_image(f)
    # fig, ax = plt.subplots()

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


    fl68_dets = fl68_detector(aligned_img, 2)

    for k, d in enumerate(fl68_dets):
        # Get the landmarks/parts for the face in box d.
        shape = fl68_sp(aligned_img, d)
        ft = Feature(shape.parts())
        if is_stroke == 1:
            stroke_features.append(ft.feature.copy())
        else:
            non_stroke_features.append(ft.feature.copy())

        # for p in shape.parts():
        #     circle = plt.Circle((p.x, p.y), 1, color="#00FF00", fill=True)
            # ax.add_patch(circle)
        # print(pose_estimate(aligned_img, shape.parts()))
        # plt.imshow(aligned_img)
        # plt.axis('off')  # Turn off axis labels
        # plt.show()

stroke_features = np.array(stroke_features)
non_stroke_features = np.array(non_stroke_features)
print("Finish feature extration")

print("Number of stroke data:", len(stroke_features))
print("Number of non stroke data:", len(non_stroke_features))

# 1 = stroke 0 = non_stroke
stroke_labels = np.array([1] * len(stroke_features))
# stroke_dataset = tf.data.Dataset.from_tensor_slices((stroke_features, stroke_labels))

non_stroke_labels = np.array([0] * len(non_stroke_features))
# non_stroke_dataset = tf.data.Dataset.from_tensor_slices((non_stroke_features, non_stroke_labels))

all_features = np.concatenate((stroke_features, non_stroke_features))
all_labels = np.concatenate((stroke_labels, non_stroke_labels))

import random
zipped_feature_labels = list(zip(all_features, all_labels))
random.shuffle(zipped_feature_labels)
all_features, all_labels = zip(*(zipped_feature_labels))

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

model = svm.SVC()
nFold = 7

kf = KFold(n_splits=nFold, shuffle=False, random_state=None)

ACC_SUM = 0
for i, (train_indices, test_indices) in enumerate(kf.split(all_features)):
    train_features = [all_features[i] for i in train_indices]
    test_features = [all_features[i] for i in test_indices]
    train_labels = [all_labels[i] for i in train_indices]
    test_labels = [all_labels[i] for i in test_indices]

    model.fit(train_features, train_labels)
    
    predicted = model.predict(test_features)

    ac = accuracy_score(predicted, test_labels)
    cm = confusion_matrix(test_labels, predicted)

    # 0 non-stroke 1 stroke
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC_SUM += ACC
    print(f"{i} Fold : TN:{TN} FN:{FN} TP:{TP} FP:{FP} Accuracy:{ACC}")
print(f"Average accuracy: {ACC_SUM / nFold}")

import joblib
# save model
joblib.dump(model, "model/svm_5_features.pkl") 



