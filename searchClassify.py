import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, time, glob, pickle
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from config import *
from lesson_functions import *

def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9,pix_per_cell=8, cell_per_block=2, hog_channel=0,
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

def processImg(iFnm, oFnm=None, saveFlev=1, imgWrt=False, dbg=False):
    if type(iFnm) == str: image  = mpimg.imread(iFnm)
    elif type(iFnm) == np.ndarray:      image = iFnm
    else: raise TypeError('Neither File Name nor Image typeError')

    imgCpy = np.copy(image)
    image  = image.astype(np.float32)/255 # conversion to 0~1 as trained on png
    heat   = np.zeros_like(image[:,:,0]).astype(np.float)

    hot_windows   = []
    searchWinList = []
    searchWinExtL = []
    for search_win in GVsearchWindows: # gVal: search windows (np.arrays(xmin,xMax, y
        # Id win coord using Modified slide_window: relative to Image
        x_start_stop = ((search_win[0][0] * image.shape[1]).round()).astype(int)
        y_start_stop = ((search_win[0][1] * image.shape[0]).round()).astype(int)
        xy_window    =  (search_win[1], search_win[1])
        searchWbin = slide_window(image, x_start_stop, y_start_stop, xy_window=xy_window)
        searchWinList += searchWbin
        searchWinExtL.append(searchWbin)

    if not imgWrt is False:
        tmpImg = np.copy(image)
        for w in range(len(searchWinExtL)):
            colorL = [0,0,0]
            colorL[w] = 225
            tmpImg = draw_boxes(tmpImg, searchWinExtL[w], color=tuple(colorL), thick=w+1)
        mpimg.imsave('/tmp/boxOut.jpg', tmpImg)

    hot_windows = search_windows(image, searchWinList, svc, X_scaler, color_space=color_space,
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

    titles  = ('OrigBoxed', 'Heat Map', 'Labels', 'Car Positions',)[-saveFlev:]
    pltImgs = (oBoxdImg, heatMap, labels[0], finnImg)[-saveFlev:]
    #pltImgs = (oBoxdImg, heatMap, (finnImg * 255).astype(np.int16))[-saveFlev:]

    if not imgWrt is False:
        for w in range(len(titles)):
            cMap=None
            if   titles[w][0] == 'H': cMap='hot'
            elif titles[w][0] == 'L': cMap='gray'
            mpimg.imsave(imgWrt+titles[w]+'.jpg', pltImgs[w], cmap=cMap)

    if dbg:
        print(titles)
        fig = plt.figure()
        for i in range(saveFlev):
            plt.subplot(100+(saveFlev+1)*10+i+1)
            plt.title(titles[i])
            if   titles[i][0] == 'H': plt.imshow(pltImgs[i], cmap='hot')
            elif titles[i][0] == 'L': plt.imshow(pltImgs[i], cmap='gray')
            else: plt.imshow(pltImgs[i])
        fig.tight_layout()
        if    oFnm != None: plt.savefig(oFnm, bbox_inches='tight')
        else: plt.savefig(imgWrt+'ALLout.png', bbox_inches='tight')
    if dbg and imgWrt == None: plt.show()
    return (finnImg * 255).astype(np.int16) # least shows Video 

def proccessVideo(inClipFnm, outClipFnm='./outPut.mp4', setBegEnd=None, setFps=12):
    if setBegEnd is None:
        print('default')
        inVclip = VideoFileClip(inClipFnm).set_fps(setFps)
    else:
        print('BegEnd')
        inVclip = VideoFileClip(inClipFnm).subclip(setBegEnd[0], setBegEnd[-1]).set_fps(setFps)
    outClip = inVclip.fl_image(processImg)
    outClip.write_videofile(outClipFnm, audio=False)

def procss6Imgs():
    for i in range(6)[:]:
        vFrame = VideoFileClip('./project_video.mp4').get_frame(30.0+i*5)
        x = processImg(vFrame, saveFlev=4, imgWrt='./output_images/'+str(i)+'_', dbg=True)
    
if __name__ == '__main__':
    if 0: createSVC(lim=0, pklIt=True)
    #inF = './project_video.mp4'; outF=outClipFnm='./PrjVideoOut.mp4'; \
        #proccessVideo(inF,outF,setFps=8,setBegEnd=None) # (17,45),
    #inF = './test_video.mp4'; outF=outClipFnm='./outPut1.mp4'; proccessVideo(inF, outF)
    #Prb: 21 (no car) 34 (2cars)
    #bboxImg = mpimg.imread('./test_images/bbox-example-image.jpg'); oFnm='./output_images/orig_1stAsIs.jpg'
    #x=processImg(bboxImg, oFnm=oFnm, saveFlev=3, dbg=True)
    #x=processImg(bboxImg, saveFlev=1, dbg=True)
    procss6Imgs()
