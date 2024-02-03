# https://stackoverflow.com/questions/12942365/detecting-a-pixelated-image-in-python

import cv2
import numpy as np

paths = ["high.jpg", "med.jpg", "none.jpg"]

for img_path in paths:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threshold_area = 100  # Adjust this threshold as needed
    pixellated_regions = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

    result_img = img.copy()
    cv2.drawContours(result_img, pixellated_regions, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{img_path[:-4]}_processed.jpg", result_img)