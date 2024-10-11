import cv2
import time
import datetime
import imutils

def motion_detection():
    video_capture = cv2.VideoCapture(0)
    time.sleep(2)

    first_frame = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        text = 'Unoccupied'

        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21, 21), 0)

        greyscale_image = gaussian_frame

        if first_frame is None:
            first_frame = greyscale_image
            continue

        frame = imutils.resize(frame, width=600)
        frame_delta = cv2.absdiff(first_frame, greyscale_image)

        thresh = cv2.threshold(frame_delta, 50, 225, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh, None, iterations=2)

        cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]

        for c in cnt:
            if cv2.contourArea(c) > 800:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = 'Occupied'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'[+] Room Status: {text}', (10, 20), font, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                    (10, frame.shape[0] - 10), font, 0.35, (0, 0, 255), 1)

        cv2.imshow('Video', frame)
        cv2.imshow('Foreground Mask', dilate_image)
        cv2.imshow('Frame Delta', frame_delta)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    motion_detection()
