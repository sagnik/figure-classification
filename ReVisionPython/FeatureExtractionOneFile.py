from skimage.io import imread,imshow
from skimage.color import rgb2grey
from skimage.transform import resize
import sys
import numpy as np
from PatchFeatureExtractionOneFile import standardizePatch, flatten_matrix, zca_whitening
import pickle
from pprint import pprint
from scipy import spatial
from datetime import datetime
import random
import os

PS=6

def extractPatch(img):
    return [img[x/125:x/125+PS,x%125:x%125+PS] for x in range((img.shape[0]-PS/2)*(img.shape[1]-PS/2))] 

def nearestOneHot(patch,cData):
    d,i=spatial.KDTree(cData).query(patch.reshape((1,PS*PS)))
    a=np.zeros(cData.shape[0])
    a[i]=1
    return a
    
      
def featureExtractOneFile(loc,clusterData,doRandom=True,randomPixno=1000):
    size=(128,128)
    im=(resize(rgb2grey(imread(loc)),size)*255) #resize
    rp=extractPatch(im)
    print "patch extraction done"
    #for each pixel that has been converted into a 36 pixel patch, we are first going to find out the nearest codebook patch. Then, 
    #we will create an one hot vector. Finally we will sum up the vectors belonging to the same quadrant and concatenate them.
    patchDict={}
    quadrant=0
    #this is too time consuming, we are choosing 5000 pixels randomly
    #randomPixelIndices=sorted(range((128-(PS/2))*(128-(PS/2))), key=os.urandom)[0:5000]
    #print len(randomPixelIndices)
    
    randomPixelIndices=range((128-(PS/2))*(128-(PS/2)))
    if doRandom: 
        random.shuffle(randomPixelIndices)
        print "shuffling done"
    else:
        randomPixno=len(randomPixelIndices)
    #for i,patch in enumerate(rp):
    for i in randomPixelIndices[:randomPixno]:
        patch=rp[i]      
        divFactor=128-(PS/2)
        x=i/divFactor
        y=i%divFactor
        if x<65 and y<65:
            quadrant=1
        elif x>65 and y<65:
            quadrant=2
        elif x>65 and y>65:
            quadrant=3
        else:
            quadrant=4
        if patch.shape[0]*patch.shape[1]==36:
            if quadrant not in patchDict:
                patchDict[quadrant]=nearestOneHot(patch,clusterData)
            else:
                patchDict[quadrant]=np.sum((patchDict[quadrant],nearestOneHot(patch,clusterData)),axis=0)
    
    #pprint(patchDict)      
    return np.hstack((patchDict[1],patchDict[2],patchDict[3],patchDict[4]))            
    
def main():
    imageLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/lines/10.1.1.182.1505-Figure-10.png"
    if len(sys.argv)==2:
        imageLoc=sys.argv[1]
    patchClusterLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/patch-clustered.nparray.pickle"  
    
    #feat=featureExtractOneFile(imageLoc,patchClusterLoc)
    clusterData=pickle.load(open(patchClusterLoc)) 
    print "cluster data loaded"
    startTime = datetime.now()
    feat=featureExtractOneFile(imageLoc,clusterData,True)

    print datetime.now() - startTime
    print feat.shape

if __name__=="__main__":
    main() 
