"""
Raw pixel intensity
Histogram of pixel intensity
Gradients of pixel intensity

templateMatching - hit/miss
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iDir = './cutouts/'
image = mpimg.imread(iDir+'bbox-example-image.jpg')
#image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)     # Make a copy of the image
    # Iterate through the bounding boxes
    for bbox in bboxes:       # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy
    
# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes for matched templates
def find_matches(img, template_list):
    nImg = np.copy(img) # Make a copy of the image to draw on
    bbox_list = []      # Define an empty list to take bbox coords
    mthd =( cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCOEFF, cv2.TM_SQDIFF,)
    for templ in template_list:
        templImg = mpimg.imread(iDir+templ)
        w, h = templImg.shape[1], templImg.shape[0]
        match = cv2.matchTemplate(nImg, templImg, mthd[0]) # use to search the image
        #using whichever of the OpenCV search methods you prefer
        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(match) #to extract the location of the best match
        # Determine bounding box corners for the match
        topL = maxLoc
        botR = topL[0]+w, topL[1]+h
        bbox_list.append((topL, botR))
        # Return the list of bounding boxes
    return bbox_list

bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
