import pickle
import numpy as np

# Original globals
color_space='YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9          # HOG orientations
pix_per_cell = 8    # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # HOG Channel 0, 1, 2, or "ALL"
spatial_size=(16,16) # Spatial binning dimensions
hist_bins = 16       # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True    # Histogram features on or off
hog_feat = True     # HOG features on or off

# Trained Classifiers
with open('./svcModel.pkl', 'rb') as fr: svc = pickle.load(fr)
with open('./X_scaler.pkl', 'rb') as rf: X_scaler = pickle.load(rf)

num_frames_to_keep = 10     # number of frames to store
recent_hot_windows = []     # list of hot windows identified on recent frames
recent_bbox_windows = []    # list of bounding boxes around cars

# classifier and training related objects
clf = None                  # classifier object
X_scaler = None             # scaler object for normalizing inputs

# hyper parameters for feature extraction
# Thresholds for procesing heatmaps
thresh_high=10
thresh_low=5

# Search widnows below indicates the areas of interest that should be searched plus the search window size.  
# The first element is the ((x_min, x_max), (y_min, y_max)) where the coordiantes are relative to the image
# size, (i.e. between 0 and 1) and the second element is the size of the search widnow:
search_window_0 = (np.array([[0.0,1.0], [0.5, 1.0]]), 32)
search_window_1 = (np.array([[0.0,1.0], [0.5, 1.0]]), 64)
search_window_2 = (np.array([[0.0,1.0], [0.5, 1.0]]), 96)
search_window_3 = (np.array([[0.0,1.0], [0.5, 1.0]]), 128)
all_search_windows = [search_window_1,
                      search_window_2, 
                      search_window_3]

# To keep track of the frame number during video processing for debugging purposes
frame_no=0                  # for debugging purposes
max_heat_list = []          # for debugging purposes
