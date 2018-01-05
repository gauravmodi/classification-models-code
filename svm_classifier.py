import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class svm_classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test  
    
    def best_fit(self, nfolds=10):
        Cs = [0.01, 0.1, 1, 10, 100]
        gammas = [0.5, 1, 2, 3, 4]
        kernels = ['rbf', 'linear']
        param_grid = {'kernel':kernels, 'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
        grid_search.fit(self.X_train, self.y_train)
        grid_search.best_params_
        return grid_search.best_params_

    # best fit for cost
    def best_fit_cost(self, fp_cost, fn_cost):
    	compare_cost = float("inf")
        kf = KFold(n_splits=10)

        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid'] 
        for kernel in kernel_list:
        	cost_kfold = list()
	        for train_index, val_index in kf.split(self.X_train):
	            X_train_kfold, X_val_kfold = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
	            y_train_kfold, y_val_kfold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
	            clf = SVC(kernel=kernel)
	            clf.fit(X_train_kfold, y_train_kfold)
	            y_pred_kfold = clf.predict(X_val_kfold)
	            c_m = confusion_matrix(y_true=y_val_kfold, y_pred=y_pred_kfold)
	            fp, fn = c_m[0,1], c_m[1, 0]
	            cost = fp_cost*fp + fn_cost*fn # Ratio of cost is 10:1
	            cost_kfold.append(cost)
	        if compare_cost > np.mean(cost_kfold):
	        	compare_cost = np.mean(cost_kfold)
	        	optimized_kernel = kernel

        print "The optimal value of kernel is: {0} for Lowest Cost = {1}".format(optimized_kernel, round(compare_cost,0))
        
    def classifier(self, kernel = 'rbf', C = 1, gamma = 'auto', fp_cost=None, fn_cost=None):
        clf = SVC(kernel=kernel, C=C, gamma=gamma)
        print '-----------------------------------------------------'
        print 'SVM CLASSIFIER'
        scores = cross_val_score(clf, self.X_train, self.y_train, cv=10)
        print '-----------------------------------------------------'
        print("Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
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
        # fp = c_m[0,1]
        # fn = c_m[1, 0]
        # cost = fp_cost*fp + fn_cost*fn # Ratio of cost is 10:1
        # print '-----------------------------------------------------'
        # print 'Cost {0}*FP + {1}*FN: {2}'.format(fp_cost, fn_cost, cost)
        # print '-----------------------------------------------------'