import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import searchClassify as sC

def hogImgPlt(i):
    Feature, hogImg = sC.single_img_features(img, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=i, spatial_feat=False,
                        hist_feat=False, hog_feat=hog_feat, hogVis=True)
    return hogImg

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#color_space = 'YCrCb'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
#hog_channel = 1
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


cars, ncar = sC.getImgFiles(lim=200)
for c in range(0,18*6,19):
    img  =  mpimg.imread(cars[c])
    oFnm = cars[c].split('/')[-1]

    fig = plt.figure()
    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.title(oFnm[:-4])
    for i in range(3):
        plt.subplot(142+i)
        plt.title('ch-'+str(i)+" "+color_space)
        plt.imshow(hogImgPlt(i), cmap='gray')
    nFnm = "/tmp/C_%s_%s.png" %(oFnm[:-4], color_space)
    plt.savefig(nFnm, bbox_inches='tight')
    #plt.show()

    img  =  mpimg.imread(ncar[c])
    oFnm = ncar[c].split('/')[-1]

    fig = plt.figure()
    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.title(oFnm[:-4])
    for i in range(3):
        plt.subplot(142+i)
        plt.title('ch-'+str(i)+" "+color_space)
        plt.imshow(hogImgPlt(i), cmap='gray')
    nFnm = "/tmp/N_%s_%s.png" %(oFnm[:-4], color_space)
    plt.savefig(nFnm, bbox_inches='tight')
    #plt.show()
