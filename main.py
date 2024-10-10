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
