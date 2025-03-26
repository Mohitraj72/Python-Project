import cv2
import numpy as np

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, img = webcam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a circular bounding box around the face
        center = (x + w // 2, y + h // 2)  # Center of the face
        radius = int(np.sqrt(w * w + h * h) // 2)  # Radius of the circle
        cv2.circle(img, center, radius, (0, 255, 0), 3)  # Draw the circle

        # Region of Interest (ROI) for eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

        # Draw rectangular bounding boxes around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Display the frame with face and eye detection
    cv2.imshow("Face and Eye Detection", img)

    # Exit the loop if the 'Esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()