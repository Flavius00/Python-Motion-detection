from encodings import normalize_encoding

import cv2
import time
import datetime
import imutils

def motion_detection():
    video_capture = cv2.VideoCapture()[0]
    time.sleep(2)

    first_frame = None
    
    while True:
        frame = video_capture.read()[1]
        text = 'Unoccupied'

        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21, 21), 0)
        blur_frame = cv2.blur(gaussian_frame, (5, 5))

        greyscale_image = blur_frame

        if first_frame is None:
            first_frame = greyscale_image
        else:
            pass

        frame = imutils.resize(frame, width=500)
        frame_delta = cv2.absdiff(first_frame, greyscale_image)

        thresh = cv2.threshold(frame_delta, 100, 225, cv2.THRESH_BINARY)[1]

        dilate_image = cv2.dilate(thresh, None, iterations=2)
        cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]



