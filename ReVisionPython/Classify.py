import pickle
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

def main():
    dataDir="../data-for-fig-classification/"
    dataPickleLoc=dataDir+"imdatawithlabels-randompixels-fast.nparray.pickle"
    datal=pickle.load(open(dataPickleLoc))
 
    accuracies=[]
    confusionmatrices=[]
    
    for i in range(10):
        np.random.shuffle(datal)
        data=datal[:,:-1]
        label=datal[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=i)
        print "starting classification, iteration: ",i+1
        y_pred=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
        a=accuracy_score(y_test, y_pred)
        c=confusion_matrix(y_test,y_pred)
        print "accuracy",a
        accuracies.append(a)
        confusionmatrices.append(c)
    
    print "average accuracy",np.mean(np.array(accuracies))
    pickle.dump(accuracies,open(dataDir+"accuracies.pickle","wb"))
    pickle.dump(confusionmatrices, open(dataDir+"confusion-matrices.pickle","wb"))    


if __name__=="__main__":
    main()
         
     
