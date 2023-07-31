import cv2
import numpy as np
import time
import torch
from PIL import Image
from torch import relu, sigmoid
from ultralytics import YOLO

from directkeys import PressKey, ReleaseKey, C

import mss


def get_data(results):
    result = results[0]
    area = result.boxes[0].xyxy[0][1] + ((result.boxes[0].xyxy[0][3] - result.boxes[0].xyxy[0][1]) / 2)
    fish = result.boxes[1].xyxy[0][1] + ((result.boxes[1].xyxy[0][3] - result.boxes[1].xyxy[0][1]) / 2)

    fish_position = fish
    area_position = area

    return fish - area


with mss.mss() as sct:
    model = YOLO('runs/detect/train39/weights/best.pt')

    last_value = None
    current_value = None


    # Part of the screen to capture
    monitor = {"top": 30, "left": 400, "width": 200, "height": 600}

    initial_time = time.time()

    counter = 0
    saving = False



    while "Screen capturing":
        last_time = time.time()
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        results = model(Image.fromarray(grayscale_img))
        value = get_data(results)
        print(f"Distance between fish and area: {value}")
        current_value = value
        if last_value is None:
            last_value = value
            continue


        ###
        ### Neural network stuff
        ###
        PressKey(C)
        time.sleep(0.01)
        ReleaseKey(C)

        ###
        ### End Neural network stuff
        ###



        last_value = value
        continue






        #res_plotted = results[0].plot()
        #cv2.imshow("result", res_plotted)

        # Display the picture
        # cv2.imshow("OpenCV/Numpy normal", img)

        # Display the picture in grayscale
        #cv2.imshow('OpenCV/Numpy grayscale',
        #            grayscale_img)

        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(resized_img, cv2.COLOR_BGRA2GRAY))

        if saving:
            print(f"Saving image {counter}")
            Image.fromarray(grayscale_img).save(f"images/img{counter}.jpg")
            counter += 1

        print(f"{torch.cuda.is_available()} - fps: {1 / (time.time() - last_time)} - time passed: {time.time() - initial_time}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # Press "s" to start saving
        if (cv2.waitKey(100) & 0xFF == ord("s")) and (not saving):
            saving = True
        elif (cv2.waitKey(100) & 0xFF == ord("s")) and saving:
            saving = False

