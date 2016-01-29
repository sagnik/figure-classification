from skimage.io import imread,imshow
from skimage.color import rgb2grey
from skimage.transform import resize
import sys
import numpy as np
from matplotlib import pyplot as plt
import pylab 
import random
from scipy import ndimage

def show_img(img):
     width = img.shape[1]/75.0
     height = img.shape[0]*width/img.shape[1]
     f = plt.figure(figsize=(width, height))
     plt.imshow(img)
     pylab.show()

def randomPatchExtraction(img):
    points=range(125*125)
    random.shuffle(points)
    
    PS=6 #patch size
    VR = 0.1 #variance ratio 
    '''
    To filter out frequently occurring constant color regions,
    we reject sample patches with variance less than 10%
    of the maximum pixel value. 	
    '''

    patchlocs=[img[x/125:x/125+PS,x%125:x%125+PS] for x in points[0:400] \
    if ndimage.variance(img[x/125:x/125+PS,x%125:x%125+PS]) > VR*ndimage.variance(img)]
    
    random.shuffle(patchlocs)
    if len(patchlocs)<=100:
        return patchlocs
    else:
        return patchlocs[0:100] 


def standardizePatch(im):
    s1=[(x-ndimage.mean(im))/ndimage.variance(im) for x in im]
    return  np.reshape(np.asarray([item for x in s1 for item in x]),(im.shape[0],im.shape[1]))     
	    
def oneImFile(loc):
    size=(128,128)
    im=(resize(rgb2grey(imread(loc)),size)*255) #resize
    #random patch extraction
    rp=randomPatchExtraction(im)
    #patch standardization
    rps=[standardizePatch(x) for x in rp]
    

    #print rps[0].shape
    #show_img(rcps[0])
    #print rcps[0]
    #print type(im),im.shape,im[5][1]
    #show_img(im)
    

def main():
    loc=sys.argv[1]
    oneImFile(loc)
    

if __name__ == "__main__":
    main()	

	
    



 
