import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import graphviz

class dtree_classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def grid_search(self, max_tree_depth, min_leaf_size):
        criterion = ['gini','entropy']
        splitter = ['best','random']
        max_depth = list(range(1,max_tree_depth))
        min_sample_leaf = list(range(1, min_leaf_size))

        param_grid = dict(max_depth=max_depth, criterion=criterion, splitter = splitter)
        scores = ['accuracy', 'recall', 'precision']
        dt = tree.DecisionTreeClassifier()
        for score in scores:
            grid_dt=GridSearchCV(dt, param_grid, cv=10,scoring=score, n_jobs= 4)
            print("Optimizing parameters for %s" % score)
            grid_dt.fit(self.X_train, self.y_train)
            print(grid_dt.best_params_)
            print(np.round(grid_dt.best_score_,3))
            
    def classifier(self, splitter, criterion, max_depth, fp_cost=1, fn_cost=1):
        dt = tree.DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=10)
        print '-----------------------------------------------------'
        print 'DECISION TREE CLASSIFIER'
        scores = cross_val_score(dt, self.X_train, self.y_train, cv=10, scoring='accuracy')
        print '-----------------------------------------------------'
        print("Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        dt.fit(self.X_train, self.y_train)
        y_pred = dt.predict(self.X_test)
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
        
    def display_tree(self):
        feature_names = X_train.columns
        dot_data = tree.export_graphviz(dt, out_file=None,
                                        feature_names=feature_names,                                        
                                        filled=True, rounded=True, 
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("dt_v1") 
        tree.export_graphviz(dt, out_file='tree.dot') 
