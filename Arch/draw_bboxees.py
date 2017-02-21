import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iDir = './cutouts/'
image = mpimg.imread(iDir+'bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(img)
    for box in bboxes: 
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img # Change this line to return image copy with boxes

# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((800, 500), (1100, 650)), 
          ((300, 500), (370, 550))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
#plt.imshow(image)
plt.show()
