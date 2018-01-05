import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class knn_classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def best_fit(self, max_k):
        cv_scores = list()

        #checking for odd values of k(neighbours) to avoid tie. 
        neighbors = [i for i in range(1,max_k) if i % 2 == 1]

        plt.figure(figsize=(10,5))

        i = 1
        for weights in ['uniform', 'distance']:
            cv_scores = list()
            for k in neighbors:
                knn= KNeighborsClassifier(n_neighbors=k, weights=weights)
            #Cross Validation
                scores = cross_val_score(knn, self.X_train, self.y_train, 
                                        cv=10, scoring='accuracy')
                cv_scores.append(scores.mean())

        # Plotting the Error rate        
            error = [1 - x for x in cv_scores]
            k_optimal = neighbors[error.index(min(error))]
            print "The optimal value for k for weight='{0}' is:\
             {1} with error rate of: {2}".format(weights, k_optimal, round(min(error),4))

            plt.subplot(1,2,i)
            plt.tight_layout()
            plt.xlabel('Number of Neighbors K')
            plt.ylabel('Misclassification Error')
            plt.title("Weight criteria: '%s')"
                      % (weights))
            plt.subplots_adjust(left=1.5, bottom=0.5, right=3.5, top=1.5)
            plt.plot(neighbors, error)
            i = i + 1
        plt.show()  
    
    # best fit for cost
    def best_fit_cost(self, fp_cost, fn_cost, max_k=15):
        cost_list = list()
        #checking for odd values of k(neighbours) to avoid tie. 
        neighbors = [i for i in range(1,max_k) if i % 2 == 1]

        plt.figure(figsize=(10,5))
        kf = KFold(n_splits=10)
   
        for k in neighbors:            
            cost_kfold = list()
            for train_index, val_index in kf.split(self.X_train):
                X_train_kfold, X_val_kfold = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
                y_train_kfold, y_val_kfold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                knn= KNeighborsClassifier(n_neighbors=k, weights='distance')
                knn.fit(X_train_kfold, y_train_kfold)
                y_pred_kfold = knn.predict(X_val_kfold)
                c_m = confusion_matrix(y_true=y_val_kfold, y_pred=y_pred_kfold)
                fp, fn = c_m[0,1], c_m[1, 0]
                cost = fp_cost*fp + fn_cost*fn # Ratio of cost is 10:1
                cost_kfold.append(cost)
            cost_list.append(np.mean(cost_kfold))

         # Plotting the Cost vs k plot      
        k_optimal = neighbors[cost_list.index(min(cost_list))]
        print "The optimal value of k is: {0} for Lowest Cost = {1}".format(k_optimal, round(min(cost_list),0))

        plt.subplot(1,1,1)
        plt.tight_layout()
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Cost')
        plt.title('Cost vs k')
        plt.subplots_adjust(left=1.5, bottom=0.5, right=3.5, top=1.5)
        plt.plot(neighbors, cost_list)
        plt.show()
        return k_optimal  
        
    def classifier(self, k, distance_criterion = 'distance', fp_cost=1, fn_cost=1):
        knn= KNeighborsClassifier(n_neighbors=k, weights='distance')
        print '-----------------------------------------------------'
        print 'k-NN CLASSIFIER'
        scores = cross_val_score(knn, self.X_train, self.y_train, cv=10, scoring='recall')
        print '-----------------------------------------------------'
        print("Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)
        print '-----------------------------------------------------'
        print 'Classification Report\n'
        print classification_report(self.y_test, y_pred, target_names=['spam', 'not spam'])
        print '-----------------------------------------------------'
        print 'Confusion Matrix\n'
        c_m = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        conf_matrix = pd.DataFrame(c_m, 
                                   columns=['Predicted Not Spam', 'Predicted Spam'],
                                   index=['Actual Not Spam', 'Actual Spam'])
        print conf_matrix
        # Calculating cost from confusion matrix
        fp = c_m[0,1]
        fn = c_m[1, 0]
        cost = fp_cost*fp + fn_cost*fn # Ratio of cost is 10:1
        print '-----------------------------------------------------'
        print 'Cost {0}*FP + {1}*FN: {2}'.format(fp_cost, fn_cost, cost)
        print '-----------------------------------------------------'