import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage
import imutils
import cv2
from skimage.feature import peak_local_max
from PIL import Image
import math

def watershed_m(im,dist):
    dist_min = dist/2 
    image = np.zeros((im.shape[0],im.shape[1],3))
    image[:,:,0] = (im * 255)
    image[:,:,1] = (im * 255)
    image[:,:,2] = (im* 255)
    image = np.uint8(image)

    gray = image[:,:,0]
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=dist,
        labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    count = 0
    average_area=list()
    xy = list()
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r < dist_min:
            cv2.circle(image, (int(x), int(y)), int(r), (255, 100,200), 1)
        else:
            cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 0), 1)
            xy.append((x,y))
            count+= 1
            average_area.append(r*r*3.14)

    if len(average_area)!=0:
        average_aera_out =sum(average_area) / len(average_area)
    else:
        average_aera_out = 0
    
    dist = 0
    dist_c = 0
    
    for combo in xy:
        min = 1000
        for combo2 in xy:
            if combo == combo2: continue
            
            if min > (math.sqrt((combo2[0]-combo[0])**2+(combo2[1]-combo[1])**2)):
                min =  (math.sqrt((combo2[0]-combo[0])**2+(combo2[1]-combo[1])**2))
        if not min == 1000:
            dist += min
            dist_c += 1
        min = 1000    
    if dist_c != 0:
        average_dist = dist/dist_c
    else:
        average_dist = 0
    return image, count, average_aera_out, average_dist      
