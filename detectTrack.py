import matplotlib.pyplot as plt
import numpy as np
import cv2, time
from config import *
from lesson_functions import slide_window
from searchClassify import search_windows
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import label
from scipy import ndimage as ndi
from moviepy.editor import VideoFileClip

def draw_bboxes_using_watershed(img, heatmap_high, heatmap_low, color=(0,0,255), thick=5, dbg=False):
    '''
    This algorithm uses watershed algorithm to put bounding boxes around cars that are close together separately
    img: original image
    heatmap_high: the heatmap that is created from on_windows using a high threshold
    heatmap_low: the heatmap that is created from on_windows using a low threshold
    color: color of the bounding box 
    thick: thickness of the bounding bix
    dbg: verbosity of the operation
    '''
    # Create masked image and calculate distance for the high thresholded heatmap using single channel
    masked_img_high = np.copy(heatmap_high[:,:,0])
    masked_img_high[masked_img_high > 0] = 1
    distance_high = ndi.distance_transform_edt(masked_img_high)  # Calculate distance from the background
    
    # Create masked image and calculate distance for the low thresholded heatmap using single channel
    masked_img_low = np.copy(heatmap_low[:,:,0])
    masked_img_low[masked_img_low > 0] = 1
    distance_low = ndi.distance_transform_edt(masked_img_low)  # Calculate distance from the background    
    
    # Calculate local maxima and convert into dtype=int
    local_maxima = peak_local_max(distance_high, indices=False, footprint=np.ones((144,144)))
    local_maxima = local_maxima.astype(int)
    # Use label function to identiry and label various local maxima that is found
    markers = label(local_maxima)[0]
    
    # Use watershed algorithm to identify various portions of the image and assume each one is a car
    labels = watershed(-distance_low, markers, mask=masked_img_low)
    
    # Identify the number of cars found
    n_cars = labels.max()
    # if dbg, print some details    
    if dbg:
        print(n_cars, ' cars found')
        cv2.imwrite(test_img_path+'pipeline_5.jpg', distance_high*255.0/distance_high.max())
        cv2.imwrite(test_img_path+'pipeline_6.jpg', distance_low*255.0/distance_low.max())        
        cv2.imwrite(test_img_path+'pipeline_7.jpg', labels*100)
    # keep a list of all bounding boxes that are drawn
    bbox_list=[]
    # Iterate through all detected cars
    for car_number in range(1, n_cars+1):
        # Find pixels with each car_number label value
        nonzero = (labels == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
        
    # Return the image
    return img, bbox_list

def draw_bboxes_using_label(img, heatmap, color=(0,0,255), thick=2, dbg=False):
    '''
    Draw bounding boxes around the cars identified in labels heatmap
    img: original image
    color: color of the bounding box 
    thick: thickness of the bounding bix
    dbg: verbosity of the operation
    '''
    # Create labels from the heatmap
    labels = label(heatmap)
    # if dbg, print some details    
    if dbg:
        print(labels[1], ' cars found')
        print('maximum intensity of heatmap = ', heatmap.max())
        cv2.imwrite(test_img_path+'pipeline_5.jpg', labels[0]*100)
    # keep a list of all bounding boxes that are drawn
    bbox_list=[]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img, bbox_list

def mark_vehicles_on_frame(frame_img, dbg=False, plot_heat_map=False, plot_box=True, watershed=True, 
                           batch_hog=True, debug=False):
    '''
    Identify the vehicles in a frame and return the revised frame with vehicles identified
    with bounding boxes
    frame_img: the frame image to be revised
    dbg: determine the verbosity of the operation
    plot_heat_map: plots the heatmap on the frame
    plot_box: plots bounding boxes on the frame
    watershed: uses the watershed algorithm for identifying cars otherwise only uses label
    batch_hog: uses batch_hog algorithm to speed up the process of hog feature extraction
    debug: debug mode
    '''
    # Define global variables
    global frame_no    
    global recent_hot_windows
    global recent_bbox_windows
    global max_heat_list
    # Identify windows that are classified as cars for all images in the recent_hot_windows
    hot_windows = []
    # Iterate through search windows that are defined globally
    for search_window in all_search_windows:
        # Identiry window coordinates using slide_window
        x_start_stop = ((search_window[0][0]*frame_img.shape[1]).round()).astype(int)
        y_start_stop = ((search_window[0][1]*frame_img.shape[0]).round()).astype(int)
        xy_window = (search_window[1], search_window[1])
        slide_windows = slide_window(frame_img.shape, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
        # Identify windows that are classified as cars                    
        hot_windows += search_windows(frame_img, search_window, slide_windows, svc, X_scaler, 
                                batch_hog=batch_hog, 
                                source_color_space=color_space, 
                                target_color_space=target_color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, 
                                hog_feat=hog_feat)
    # if dbg, save some photos    
    if dbg:
        pipeline_1 = np.copy(frame_img)
        for window in hot_windows:
            cv2.rectangle(pipeline_1, window[0], window[1], color=(0,0,255), thickness=2)
        cv2.imwrite(test_img_path+'pipeline_1.jpg', pipeline_1) 
    
    # Append the results to the global list
    recent_hot_windows.append(hot_windows)
    if len(recent_hot_windows) > num_frames_to_keep:
        recent_hot_windows.pop(0)
    # Create heatmap from the hot_windows
    heatmap = np.zeros_like(frame_img)
    for frame_hot_windows in recent_hot_windows:
        for window in frame_hot_windows:
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    # Normalize the pixel values
    heatmap = cv2.convertScaleAbs(heatmap, heatmap, 1/len(recent_hot_windows))    
    
    # if dbg, plot the heatmap and save to file   
    if dbg:
        scaled_heatmap = np.copy(heatmap)
        scaled_heatmap = cv2.convertScaleAbs(heatmap,scaled_heatmap,255/heatmap.max())
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_2.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap, 0.7, 0))
    
    # Zero out pixels below the threshold and construct both high and low thresholded heatmaps
    heatmap_high = np.copy(heatmap)
    heatmap_high[heatmap_high <= thresh_high] = 0    
    heatmap_low = np.copy(heatmap)
    heatmap_low[heatmap_low <= thresh_low] = 0     
    # if dbg, save some photos and print some details   
    if dbg:
        print('maximum intensity of heatmap_high = ', heatmap_high.max())
        # save heatmap_high
        scaled_heatmap_high = np.copy(heatmap_high)
        scaled_heatmap_high = cv2.convertScaleAbs(heatmap_high,scaled_heatmap_high,255/heatmap_high.max())
        scaled_heatmap_high[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_3.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap_high, 0.7, 0))
        # save heatmap_low
        scaled_heatmap_low = np.copy(heatmap_low)
        scaled_heatmap_low = cv2.convertScaleAbs(heatmap_low,scaled_heatmap_low,255/heatmap_low.max())
        scaled_heatmap_low[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_4.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap_low, 0.7, 0))
    
    # Draw the bounding boxes on the images
    draw_image = np.copy(frame_img)
    if plot_box:
        draw_color = [0,0,0]
        draw_color[color_space.index('B')] = 255
        draw_color = tuple(draw_color)
        if watershed:
            draw_image, bbox_list = draw_bboxes_using_watershed(draw_image, heatmap_high, heatmap_low, 
                                                                color=draw_color, thick=1, dbg=dbg) 
        else:
            draw_image, bbox_list = draw_bboxes_using_label(draw_image, heatmap_high, 
                                                            color=draw_color, thick=1, dbg=dbg) 
    
    # keep track of the bounding goxes in the most recent frames
    recent_bbox_windows.append(bbox_list)
    if len(recent_bbox_windows) > num_frames_to_keep:
        recent_bbox_windows.pop(0)
    
    # plot heatmap on frame
    if plot_heat_map:
        scaled_heatmap_low = heatmap_low*100
        scaled_heatmap_low[scaled_heatmap_low>255] = 255
        scaled_heatmap_low[:,:,:2] = 0
        draw_image = cv2.addWeighted(draw_image, 1, scaled_heatmap_low, 0.5, 0)
    # Save individual frames for debugging purposes if required
    if debug:
        print(heatmap.max())
        scaled_heatmap = np.copy(heatmap)
        scaled_heatmap = cv2.convertScaleAbs(heatmap,scaled_heatmap,255/heatmap.max())
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(work_path+'tmp/frame_{:04d}.png'.format(frame_no), 
                    cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(work_path+'tmp/heatmap_{:04d}.jpg'.format(frame_no), 
                    cv2.cvtColor(cv2.addWeighted(np.copy(draw_image), 1, scaled_heatmap, 0.7, 0),cv2.COLOR_RGB2BGR))
    frame_no+=1
    return draw_image

def process_test_images(sequence=False, dbg=False, 
                        high_threshold=10, low_threshold=5, watershed=True, batch_hog=True):
    '''
    Read test images, process them, mark the vehicles on them and save them back to the folder
    sequence: If an image part of sequence of frames, then recent_hot_windows will be updated as if movie stream
    high_threshold: the value of the high_threshold
    low_threshold: the value of the low threshold
    watershed: if True uses the watershed algorithm for identifying cars otherwise only uses label
    batch_hog: if True uses a faster HOG extraction algorithm
    '''
    global recent_hot_windows
    global thresh_high
    global thresh_low
    global num_frames_to_keep
    thresh_high = high_threshold
    thresh_low = low_threshold
    
    # Read test images and show search rectanbles on them
    file_formats = ['*.jpg', '*.png', '*.jpeg']
    # Iterate through files
    for file_format in file_formats:
        file_names = glob.glob(test_img_path+file_format)
        for file_name_from in file_names:
            # Load image
            img = cv2.imread(file_name_from) 
            # Recorde time if dbg = True
            if dbg:
                t_start = time.time()
            # recent rect_hot_windows each time if not processing sequnce images
            if not sequence: recent_hot_windows = []
            # process image
            img_rev = mark_vehicles_on_frame(img, dbg=dbg, watershed=watershed, batch_hog=batch_hog)
            # Recorde time and print details if dbg = True
            if dbg:
                t_finish = time.time()
                print('Total time for ', os.path.basename(file_name_from), ': ', 
                      round(t_finish-t_start, 2), 'Seconds')
            # Save image to file
            file_name_to = 'processed_'+os.path.basename(file_name_from)
            cv2.imwrite(test_img_path+file_name_to, img_rev) 

if __name__ == '__main__':
    x = 1
