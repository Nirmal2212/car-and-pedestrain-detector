import cv2
import numpy as np

# Create our body classifier (pre-trained classifier)
body_classifier = cv2.CascadeClassifier('D:\EE\car detection\data_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('D:\EE\car detection\walking.avi')

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()





# import cv2
# import time
# import numpy as np

# # Create our body classifier
# car_classifier = cv2.CascadeClassifier('D:\EE\car detection\data_car.xml')

# # Initiate video capture for video file
# cap = cv2.VideoCapture('D:\EE\car detection\car1.mp4')


# # Loop once video is successfully loaded
# while cap.isOpened():
    
#     time.sleep(.05)
#     # Read first frame
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
#     # Pass frame to our car classifier
#     cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
#     # Extract bounding boxes for any bodies identified
#     for (x,y,w,h) in cars:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#         cv2.imshow('Cars', frame)

#     if cv2.waitKey(1) == 13: #13 is the Enter Key
#         break

# cap.release()
# cv2.destroyAllWindows()