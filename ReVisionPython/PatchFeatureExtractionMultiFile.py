from PatchFeatureExtractionOneFile import patchFeatExtractionOneImFile
import sys,os
import pickle
import numpy as np

def main():
   
    #default locations
    patchImDir="/home/sagnik/codes/figure-classification/data-for-fig-classification/patchimages/"
    pickleLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/nonzcapatch-unclustered.nparray.pickle"

    if len(sys.argv)==3:
        patchImDir=sys.argv[1]
        pickleLoc=sys.argv[2]
    elif len(sys.argv)==2:
        patchImDir=sys.argv[1]
        pickleLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/nonzcapatch-unclustered.nparray.pickle"
  
    imFiles=[patchImDir+x for x in os.listdir(patchImDir) if x.endswith("png")]
    a=np.zeros(36).reshape((1,36))
    
    for index,f in enumerate(imFiles):
        print "processing: ",index+1,"out of",len(imFiles),f
        a=np.vstack((a,patchFeatExtractionOneImFile(f)))
    
    data=a[1:,:]
    
    print "writing data"
    pickle.dump(data,open(pickleLoc,"wb"))    

if __name__ == "__main__":
    main()

