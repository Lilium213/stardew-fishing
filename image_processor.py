import cv2
import numpy
import time
import torch
from ultralytics import YOLO

import mss

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

    initial_time = time.time()

    while "Screen capturing":
        last_time = time.time()
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        resized_img = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)

        # Display the picture
        # cv2.imshow("OpenCV/Numpy normal", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        cv2.imshow('OpenCV/Numpy grayscale',
                   cv2.cvtColor(resized_img, cv2.COLOR_BGRA2GRAY))

        print(f"{torch.cuda.is_available()} - fps: {1 / (time.time() - last_time)} - time passed: {time.time() - initial_time}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break