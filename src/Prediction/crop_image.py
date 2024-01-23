import cv2
import copy

def crop_face(frame):

    frame_mutable = copy.deepcopy(frame)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Margin around the detected face for cropping
    margin = 25

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame_mutable, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        # Expand the region of interest (ROI) by adding a margin
        x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin

        # Ensure the coordinates are non-negative
        x = max(0, x)
        y = max(0, y)

        # Draw a blue rectangle around the detected face in the original frame
        cv2.rectangle(frame_mutable, (x - 25, y - 25), (x + w + 25, y + h + 25), (0, 128, 128), 2)  # Dim yellow

        # Crop the detected face with margin
        face_crop = frame_mutable[y:y+h, x:x+w]

        # Return both the original frame with the blue square and the cropped face
        return frame_mutable, face_crop

    # If no faces are detected, return the original frame without modifications
    return frame, frame
