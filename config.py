import cv2, pickle
import numpy as np
from skimage.feature import hog
from lesson_functions import bin_spatial, color_hist

# Original globals
color_space='YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
source_color_space='RGB'
target_color_space='YCrCb'
orient         = 9  # HOG orientations
pix_per_cell   = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel  = "ALL" # HOG Channel 0, 1, 2, or "ALL"
spatial_size=(16,16) # Spatial binning dimensions
hist_bins    = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat    = True # Histogram features on or offy
hog_feat     = True # HOG features on or off

# Retrieve pickled Trained Classifiers
with open('./svcModel.pkl', 'rb') as fr: svc = pickle.load(fr)
with open('./X_scaler.pkl', 'rb') as rf: X_scaler = pickle.load(rf)

num_frames_to_keep  = 10 # no. of frames to keep
recent_hot_windows  = [] # list of hot windows identified on recent frames
recent_bbox_windows = [] # list of bounding boxes around cars

# Thresholds for heatmaps
thresh_high = 10
thresh_low  = 5
base_size   = 64

# searchWidnows: Areas of interest to be searched. The elements are :
#((xmin, xMax), (ymin, yMax)) 
# the coordiantes are relative to the image size, (i.e. between 0 and 1)
# search widnow size
GVsearchWindows = [ (np.array([[0.0,1.0], [0.5, 1.0]]), 64), 
                  (np.array([[0.0,1.0], [0.5, 1.0]]), 96),
                  (np.array([[0.0,1.0], [0.5, 1.0]]), 128),]

# dbg var: Frame Tracking while VideoFileClip Processing
frame_no = 0       # dbg var
max_heat_list = [] # dbg var
