import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import glob

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgFs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    features = []     # Create a list to append feature vectors to
    for fileNm in imgFs:
        img = mpimg.imread(fileNm)
        if cspace != 'RGB':
            if   cspace == 'HSV': featureImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV': featureImg = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS': featureImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV': featureImg = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else: featureImg = np.copy(img)
        featuresSpatial  = bin_spatial(featureImg, size=spatial_size)
        featuresColorHst = color_hist( featureImg, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((featuresSpatial, featuresColorHst)))
    return features

images  = glob.glob('*.jpeg')
cars    = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image: notcars.append(image)
    else:                                       cars.append(image)
        
car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),hist_bins=32,hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),hist_bins=32,hist_range=(0, 256))

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    rand_state = np.random.randint(0, len(cars))
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
    print('For labels: ', y_test[0:10])

    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else: 
    print('Your function only returns empty feature vectors...')

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
X 
