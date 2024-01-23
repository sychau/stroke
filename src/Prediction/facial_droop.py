import joblib
import dlib
from sklearn import svm
from face_feature import Feature
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

class Predictor:
	model = None
	predictor_fl5_path = ""
	predictor_fl68_path = ""
	fl5_detector = None
	fl5_sp = None
	fl68_detector = None
	fl68_sp = None

	def __init__(self, model_path) -> None:
		self.model = joblib.load(model_path)

		self.predictor_fl5_path = "model/shape_predictor_5_face_landmarks.dat"
		self.predictor_fl68_path = "model/shape_predictor_68_face_landmarks.dat"

		self.fl5_detector = dlib.get_frontal_face_detector()
		self.fl5_sp = dlib.shape_predictor(self.predictor_fl5_path)

		self.fl68_detector = dlib.get_frontal_face_detector()
		self.fl68_sp = dlib.shape_predictor(self.predictor_fl68_path)

	# return 1 if stroke found in image, return 0 if no stroke found in image. Returns 42 if no face was found..
	# img must be an 3D array in RGB mode
	def predict(self, img):
		fl5_dets = self.fl5_detector(img, 1)
		num_faces = len(fl5_dets)
		if num_faces == 0:
			# print("Sorry, there were no faces found in this image")
			return 42
			# exit or warning or return 0?
			
		faces = dlib.full_object_detections()

		for detection in fl5_dets:
			faces.append(self.fl5_sp(img, detection))

		# assume only 1 face in image 
		aligned_img = dlib.get_face_chip(img, faces[0])
		fl68_dets = self.fl68_detector(aligned_img, 2)

		feature_5 = []
		for k, d in enumerate(fl68_dets):
        	# Get the landmarks/parts for the face in box d.
			shape = self.fl68_sp(aligned_img, d)
			facing_angle = Predictor.pose_estimate(aligned_img, shape.parts())

			angle_threshhold = 15
			if abs(facing_angle[1] > angle_threshhold) or abs(facing_angle[2] > angle_threshhold):
				print("not facing front")
				return 42
			if len(shape.parts()) != 68: # landmarks not constructed
				return 42
			ft = Feature(shape.parts())
			feature_5 = ft.feature
		if len(feature_5) != 5:
			return 42
		result = self.model.predict([feature_5])
		return result[0]

	@staticmethod
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

# Example usage
def test():
	pd = Predictor("model/svm_5_features.pkl")
	img = dlib.load_rgb_image("dataset/non_stroke/a8.jpg")

	print(pd.predict(img))

if __name__ == "__main__":
    test()