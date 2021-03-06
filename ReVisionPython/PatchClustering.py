from sklearn.cluster import KMeans
import pickle
import sys
import numpy as np
from scipy import stats
from sklearn import metrics


def evaluateImbalanced(true,pred):
    if len(true)!=len(pred):
        print "true and predicted labels length different"
        return 0
    else:
        return 1 #TODO: this needs to be explored more         

def main():
    #default locations
    dataLoc="../data-for-fig-classification/zcapatch-unclustered-labeled.nparray.pickle"
    saveLoc="../data-for-fig-classification/zcapatch-clustered.nparray.pickle"
    '''   
    if len(sys.argv)==3:
        dataLoc=sys.argv[1]
        saveLoc=sys.argv[2]
    elif len(sys.argv)==2:
        dataLoc=sys.argv[1]
        saveLoc="../data-for-fig-classification/nonzcapatch-clustered.nparray.pickle"
    '''
    datal=pickle.load(open(dataLoc))
    np.random.shuffle(datal)

    data=datal[:,:-1]
    l_true=datal[:,-1]

    NOC=200 #no. of clusters, default 200
    print "clustering ",data.shape[0],"data points in",NOC,"clusters"
    
    km = KMeans(n_clusters=NOC, init='k-means++',random_state=1).fit(data)
    labels = km.labels_
    
    l_pred=labels 
    print "clustering done"  

    #TODO: for now, let's just assume that zca actually helps, because when NOC=3, zca shows almost 100% imporvement. 
    #But an important question remains: how to evaluate a code book?
    
    #evaluateImbalanced(l_true,l_pred)
    #print "ARI",metrics.adjusted_rand_score(l_true, l_pred)
    #print "HSCore",metrics.homogeneity_score(l_true,l_pred)    
    
    #TODO: there are some highly frequent clusters, possibly from white pixels. Should we remove them?
    #print stats.itemfreq(labels) 
     
    labels=list(labels)
    #print len(labels)
    #labels=[0]*data.shape[0]
    
    clusterDict={}
    for i,v in enumerate(labels):
        #print i
        if v not in clusterDict:
            clusterDict[v]=data[i].reshape(1,36)
        else:      
            clusterDict[v]=np.vstack((clusterDict[v],data[i].reshape(1,36)))
    
    clusterMeans=np.zeros(36).reshape((1,36))

    for x in clusterDict.keys():
        print x,clusterDict[x].shape
        clusterMeans=np.vstack((clusterMeans,np.sum(clusterDict[x],axis=0)/clusterDict[x].shape[0]))
    
    clusterMeans=clusterMeans[1:,:]
    print clusterMeans.shape
    
    pickle.dump(clusterMeans,open(saveLoc,"wb"))
    
if __name__=="__main__":
    main()    
        
