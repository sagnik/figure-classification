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
from scipy import ndimage

PS=6
VR = 0.1

def extractPatch(img):
    return [img[x/125:x/125+PS,x%125:x%125+PS] for x in range((img.shape[0]-PS/2)*(img.shape[1]-PS/2))] 

def extractPatch1(img):
    return filter(lambda y: y.shape[0]*y.shape[1]==PS*PS, [standardizePatch(img[x/64:x/64+PS,x%64:x%64+PS]) for x in range(64*64) \
    if ndimage.variance(img[x/64:x/64+PS,x%64:x%64+PS]) > VR*ndimage.variance(img)])

def nearestOneHot(patch,cData):
    d,i=spatial.KDTree(cData).query(patch.reshape((1,PS*PS)))
    a=np.zeros(cData.shape[0])
    a[i]=1
    return a
    
def standardizePatch(im):
    s1=[(x-ndimage.mean(im))/(ndimage.variance(im)+0.01) for x in im]
    return  np.reshape(np.asarray([item for x in s1 for item in x]),(im.shape[0],im.shape[1]))     

def get2DMatrix(patchList):
    a=np.zeros(36).reshape((1,36))
    for p in patchList:
        a=np.vstack((a,flatten_matrix(p)))
    return a[1:,:]

def patchResponseMap(quadrant,clusterData):
    clusterData=normalize(clusterData,1)
    quadrant=normalize(quadrant,1)
    maxCosines=list(np.argmax(np.dot(quadrant,clusterData.T),1))
    #print quadrant.shape,clusterData.T.shape,len(maxCosines)
    a=np.zeros(200).reshape((1,200))
    for index in maxCosines:
        b=[0]*200
        b[index]=1
        a=np.vstack((a,np.array(b).reshape((1,200))))
    
    #print a.shape
    return np.sum(a[1:,:],axis=0) 
    #return np.sum(np.dot(quadrant,clusterData.T),axis=0).reshape(1,200)
   
    

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def featureExtractOneFileUnit(loc,clusterData):
    size=(128,128)
    im=(resize(rgb2grey(imread(loc)),size)*255) #resize

    rpq1=patchResponseMap(get2DMatrix(extractPatch1(im[0:64,0:64])),clusterData)
    rpq2=patchResponseMap(get2DMatrix(extractPatch1(im[64:,0:64])),clusterData)
    rpq3=patchResponseMap(get2DMatrix(extractPatch1(im[64:,64:])),clusterData)
    rpq4=patchResponseMap(get2DMatrix(extractPatch1(im[0:64,64:])),clusterData)

    #print "patch extraction done",rpq1.shape,rpq2.shape,rpq3.shape,rpq4.shape
    	

    return np.hstack((rpq1,rpq2,rpq3,rpq4))            
      

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
            patch=standardizePatch(patch)
            if quadrant not in patchDict:
                patchDict[quadrant]=nearestOneHot(patch,clusterData)
            else:
                patchDict[quadrant]=np.sum((patchDict[quadrant],nearestOneHot(patch,clusterData)),axis=0)
    
    #pprint(patchDict)      
    return np.hstack((patchDict[1],patchDict[2],patchDict[3],patchDict[4]))            

    
def main():
    imageLoc="../data-for-fig-classification/lines/10.1.1.182.1505-Figure-10.png"
    if len(sys.argv)==2:
        imageLoc=sys.argv[1]
    patchClusterLoc="../data-for-fig-classification/zcapatch-clustered.nparray.pickle"  
    
    clusterData=pickle.load(open(patchClusterLoc)) 
    print "cluster data loaded"
 
    startTime = datetime.now()
 
    feat=featureExtractOneFile(imageLoc,clusterData,doRandom=False)
     #feat=featureExtractOneFileUnit(imageLoc,clusterData)

    print datetime.now() - startTime
    print feat.shape

if __name__=="__main__":
    main() 
