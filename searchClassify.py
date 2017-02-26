import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time, glob, pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from config import *

def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hogVis=False, dbg=False):
    #1) Define an empty list to receive features
    img_features, imgShapes = [], {}
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if   color_space == 'HSV':  feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':  feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':  feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':  feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
        if dbg: imgShapes["spatial"]=spatial_features.shape[0]
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
        if dbg: imgShapes["histgrm"]=hist_features.shape[0]
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
                if dbg: imgShapes['HOG'+str(channel)] = len(hog_features)
        else:
            if hogVis: return get_hog_features(feature_image[:,:,hog_channel], orient,
                                               pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                if dbg: imgShapes['HOG0'] = len(hog_features)
        #8) Append features to list
        img_features.append(hog_features)
    #9) Return concatenated array of features
    if dbg: print(imgShapes)
    return np.concatenate(img_features)

def extractFeatures(imgFileNms, color_space='YCrCb', spatial_size=(32, 32),
                    hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgFileNms:
        img   =  mpimg.imread(file)
        fileFeature = single_img_features(img, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(fileFeature)
    return features

def search_windows(img, windows, clf, scaler, color_space='YCrCb',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True, dbg=False):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1: on_windows.append(window)
        if dbg and prediction==1: print('on_windows.append', len(on_windows))
    #8) Return windows for positive detections
    return on_windows

def getImgFiles(lim=0):
    images = []
    for subDir in glob.glob('./non-vehicles/*'):
        images = images + glob.glob(subDir+'/*.png')
    for subDir in glob.glob('./vehicles/*'):
        images = images + glob.glob(subDir+'/*.png')
    carS, notcars = [], []
    for image in images:
        if 'non-vehicles' in image: notcars.append(image)
        else:                          carS.append(image)
    if lim > 1: carS, notcars = carS[:lim], notcars[:lim]
    print('# of carS example = ', len(carS))
    print('# of not cars are = ', len(notcars))
    return carS, notcars

def createSVC(lim=0, pklIt=False):
    cars, notCars = getImgFiles(lim=lim)
    car_features = extractFeatures(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extractFeatures(notCars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)     # Fit a per-column scaler
    scaled_X = X_scaler.transform(X)       # Apply the scaler to X
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    if pklIt:
        with open('./svcModel.pkl', 'wb') as fp: pickle.dump(svc,fp)
        with open('./X_scaler.pkl', 'wb') as fw: pickle.dump(X_scaler, fw)


def processImg(iFnm, oFnm=None, saveFlev=1, dbg=False):
    if type(iFnm) == str: image  = mpimg.imread(iFnm)
    elif type(iFnm) == np.ndarray:      image = iFnm
    else: raise TypeError('Neither File Name nor Image typeError')

    imgCpy = np.copy(image)
    image  = image.astype(np.float32)/255 # conversion to 0~1 as trained on png
    heat   = np.zeros_like(image[:,:,0]).astype(np.float)
    searchWindows = slide_window(image, x_start_stop=[None, None], y_start_stop = [475, None],
                                 xy_window=(96, 96), xy_overlap=(0.5, 0.5), dbg=dbg)
    hot_windows   = search_windows(image, searchWindows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat, dbg=dbg)

    oBoxdImg = draw_boxes(imgCpy, hot_windows, color=(0,0,255), thick=6)
    heatAdd  = add_heat(heat, hot_windows)
    heatMap  = np.clip(heatAdd, 0, 255)
    labels   = label(heatMap)
    finnImg  = draw_labeled_bboxes(np.copy(image), labels)

    titles  = ('OrigBoxed', 'Heat Map', 'Car Positions',)[-saveFlev:]
    pltImgs = (oBoxdImg, heatMap, finnImg)[-saveFlev:]
    #pltImgs = (oBoxdImg, heatMap, (finnImg * 255).astype(np.int16))[-saveFlev:]
    if dbg:
        print(titles)
        fig = plt.figure()
        for i in range(saveFlev):
            plt.subplot(100+(saveFlev+1)*10+i+1)
            plt.title(titles[i])
            if titles[i][:4] != 'Heat': plt.imshow(pltImgs[i])
            else: plt.imshow(pltImgs[i], cmap='hot')
        fig.tight_layout()
        if oFnm != None: plt.savefig(oFnm, bbox_inches='tight')
        if dbg: plt.show()
    return (finnImg * 255).astype(np.int16) # least shows Video 

def proccessVideo(inClipFnm, outClipFnm='./outPut.mp4'):
    inVclip = VideoFileClip(inClipFnm)
    outClip = inVclip.fl_image(processImg)
    outClip.write_videofile(outClipFnm, audio=False)


if __name__ == '__main__':
    if 0: createSVC(lim=0, pklIt=True)
    bboxImg = mpimg.imread('./test_images/bbox-example-image.jpg'); oFnm='./output_images/orig_1stAsIs.jpg'
    #x = processImg(bboxImg, oFnm=oFnm, saveFlev=3, dbg=True)
    x = processImg(bboxImg, saveFlev=1, dbg=True)
    vFrame = VideoFileClip('./test_video.mp4').get_frame(1.0); x = processImg(vFrame,  saveFlev=3, dbg=True)
    #inF = './test_video.mp4'; outF=outClipFnm='./outPut1.mp4'; proccessVideo(inF, outF)
    #inF = './project_video.mp4'; outF=outClipFnm='./PrjVideoOut.mp4'; proccessVideo(inF, outF)
    #bboxImg.shape;    vFrame.shape; Out[17]: (720, 1280, 3)
