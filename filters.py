import cv2
import numpy as np

cap = cv2.VideoCapture('logo_wentzl_cam_be5c6c.stream-11_30-2017-12fps.mp4.mp4')

if cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("Frame", frame)
        #bilateral = cv2.Sobel(frame, cv2.CV_64F, 2, 0, ksize=5)
        bilateral = cv2.Laplacian(frame, cv2.CV_32F)
        cv2.imshow("Bilateral", bilateral)

        while True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
