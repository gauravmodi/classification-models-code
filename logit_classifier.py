import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class logit_classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def best_fit(self, c_list=(100, 1, 0.01), fp_cost=10, fn_cost=1):
        # cost_list_l1 = list()
        # cost_list_l2 = list()
        compare_cost = float("inf")
        penatly = 'L1'
        kf = KFold(n_splits=10)
        for i, C in enumerate(c_list):
    # turn down tolerance for short training time
            clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
            clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
            clf_l1_LR.fit(self.X_train, self.y_train)
            clf_l2_LR.fit(self.X_train, self.y_train)
            cost_l1_kfold = list()
            cost_l2_kfold = list()
            for train_index, val_index in kf.split(self.X_train):
                X_train_kfold, X_val_kfold = self.X_train.iloc[train_index, :], self.X_train.iloc[val_index, :]
                y_train_kfold, y_val_kfold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
                clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
                clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
                clf_l1_LR.fit(X_train_kfold, y_train_kfold)
                clf_l2_LR.fit(X_train_kfold, y_train_kfold)
                # knn= KNeighborsClassifier(n_neighbors=k, weights='distance')
                # knn.fit(X_train_kfold, y_train_kfold)
                y_pred_l1_kfold = clf_l1_LR.predict(X_val_kfold)
                y_pred_l2_kfold = clf_l2_LR.predict(X_val_kfold)
                c_m_l1 = confusion_matrix(y_true=y_val_kfold, y_pred=y_pred_l1_kfold)
                c_m_l2 = confusion_matrix(y_true=y_val_kfold, y_pred=y_pred_l2_kfold)
                fp_l1, fn_l1 = c_m_l1[0,1], c_m_l1[1, 0]
                fp_l2, fn_l2 = c_m_l2[0,1], c_m_l2[1, 0]
                cost_l1 = fp_cost*fp_l1 + fn_cost*fn_l1 # Ratio of cost is 10:1
                cost_l2 = fp_cost*fp_l2 + fn_cost*fn_l2 # Ratio of cost is 10:1
                cost_l1_kfold.append(cost_l1)
                cost_l2_kfold.append(cost_l2)
            if min(np.mean(cost_l1_kfold), np.mean(cost_l1_kfold)) < compare_cost:
                compare_cost = min(np.mean(cost_l1_kfold), np.mean(cost_l1_kfold))
                optimize_C = C
                if np.mean(cost_l1_kfold) < np.mean(cost_l2_kfold):
                    penalty = 'L1'
                else:
                    penalty = 'L2'

        print 'Best parameters are\nPenalty={0} and C={1}'.format(penalty, optimize_C)
            # cost_list_l1.append(np.mean(cost_l1_kfold))
            # cost_list_l2.append(np.mean(cost_l2_kfold))

            
    def classifier(self, penalty='l1', C=1, multi_class='ovr', fp_cost=1, fn_cost=1):
        logit = LogisticRegression(penalty=penalty, C=C, multi_class=multi_class)
        print '-----------------------------------------------------'
        print 'LOGISTIC REGRESSION CLASSIFIER'
        scores = cross_val_score(logit, self.X_train, self.y_train, cv=10, scoring='accuracy')
        print '-----------------------------------------------------'
        print("Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        logit.fit(self.X_train, self.y_train)
        y_pred = logit.predict(self.X_test)
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