"""
# Build Models and Export Predictions and Probabilities

Base Models: Random Forest, KNN, SVC 

Blending Ensemble Model using Extreme Gradient Boost 

"""

## Data Manipulation
import numpy as np
import pandas as pd

## Visualization
import matplotlib.pyplot as plt

## Data Engineering
from sklearn.model_selection import train_test_split

## Model Building
from sklearn.ensemble import RandomForestClassifier # Base Model 1
from sklearn.neighbors import KNeighborsClassifier # Base Model 2
from sklearn.svm import SVC # Base Model 3
from xgboost import XGBClassifier # Meta Model 

## Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

## Other
from tqdm import tqdm # Python Progress Bars
import os

## Performance Evaluation 
import quantstats as qs

## Import Functions
from functions import *



#__________________________________________________________________________________
### Class: Export Performance Results from Selected Models

class ModelEval:

    def __init__(self, X, y, basemodels, testsize):
        self.X = X
        self.y = y
        self.models = basemodels
        self.testsize = testsize

    ## Accuracy score train      
    def get_accuracy_train(self, ypred):
        score = accuracy_score(self.y_train, ypred)
        print(f'Classifier Accuracy Train: {score*100:.4}')

    ## Accuracy score val    
    def get_accuracy_val(self, ypred):
        score = accuracy_score(self.y_val, ypred)
        print(f'Classifier Accuracy Train: {score*100:.4}')


    ## Accuracy score test
    def get_accuracy_test(self, ypred):
        score = accuracy_score(self.y_test, ypred)
        print(f'Classifier Accuracy Test: {score*100:.4}')


    ## Classification Report
    def get_classification_report(self, ypred):
        print("Classification Report:")
        print(classification_report(self.y_test, ypred))

  
    def plot_confusion_matrix_and_roc(self, ypred, yprob, savename):
        
        ## Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ## Plot Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(self.y_test, ypred, cmap=plt.cm.Blues, ax=ax1)
        ax1.set_title("Confusion Matrix")

        ## Plot ROC Curve
        RocCurveDisplay.from_predictions(self.y_test, yprob, ax=ax2)
        ax2.plot([0,1], [0,1], linestyle="--", label='Random 50:50')
        ax2.legend()
        ax2.set_title("AUC-ROC Curve")

        ## Adjust Layout
        plt.tight_layout()

        ## Save Plots        
        os.makedirs(os.path.dirname(savename), exist_ok=True)  # Create directories if they don't exist
        plt.savefig(savename, format='pdf', bbox_inches='tight')

        # Show plots
        plt.show()

#__________________________________________________________________________________



#__________________________________________________________________________________

### Class: Randon Forest Model
    
class RFModel(ModelEval):

    def __init__(self, X, y, basemodels=None, testsize=0.15, hyperparam=None): 
        
        ## Call Superclass - initialize attributes from the parent class
        super().__init__(X, y, basemodels, testsize)

        ## Store the hyperparameters array
        self.hyperparam = hyperparam

        
        ## Split Data - Train and Test
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(self.X, 
                                                                                 self.y, 
                                                                                 random_state=55,
                                                                                 test_size=self.testsize, 
                                                                                 shuffle=False)

    ## Compute Prediction and Probability
    def fit_model(self):

        ## Intent Model 
        model = RandomForestClassifier( n_jobs=-1, random_state=55, 
                                        class_weight= cwts(self.y), # Balance Class
                                        n_estimators=self.hyperparam['n_estimators'], # Set the number of trees (estimators).  
                                        max_depth=self.hyperparam['max_depth'], # Controls the complexity of each tree. 
                                        max_features = self.hyperparam['max_features'], 
                                        min_samples_leaf=1,  # Minimum number of samples in a leaf node
                                        min_samples_split=2, # Min samples required to split an internal node in a decision tree
                                        criterion='gini'  # Use Gini impurity as the split criterion. 
                                    )
        ## Trai/Fit Model
        model.fit(self.X_train, self.y_train)

        ## Predic Model - Train
        ypred_train = model.predict(self.X_train)
        
        ## Predic Model - Test
        ypred = model.predict(self.X_test)

        ## Probability Model
        yprob = model.predict_proba(self.X_test)[:,1]

        return ypred_train, ypred, yprob



#__________________________________________________________________________________
    
### Class: KNN 
    
class KnnModel(ModelEval):

    def __init__(self, X, y, basemodels=None, testsize=0.15, hyperparam=None): 
        
        ## Call Superclass - initialize attributes from the parent class
        super().__init__(X, y, basemodels, testsize)

        ## Store the hyperparameters array
        self.hyperparam = hyperparam

        ## Split Data - Train and Test
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(self.X, 
                                                                                 self.y, 
                                                                                 random_state=55,
                                                                                 test_size=self.testsize, 
                                                                                 shuffle=False)

    ## Compute Prediction and Probability
    def fit_model(self):

        ## Intent Model 
        model = KNeighborsClassifier( n_jobs=-1, 
                                      n_neighbors = self.hyperparam['n_neighbors']
                                    )
        ## Trai/Fit Model
        model.fit(self.X_train, self.y_train)

        ## Predic Model - Train Data
        ypred_train = model.predict(self.X_train)
        
        ## Predic Model - Test Data
        ypred = model.predict(self.X_test)

        ## Probability Model
        yprob = model.predict_proba(self.X_test)[:,1]
        

        return ypred_train, ypred, yprob



#__________________________________________________________________________________
    
### Class: SVC 
    
class SvcModel(ModelEval):

    def __init__(self, X, y, basemodels=None, testsize=0.15, hyperparam=None): 
        
        ## Call Superclass - initialize attributes from the parent class
        super().__init__(X, y, basemodels, testsize)

        ## Store the hyperparameters array
        self.hyperparam = hyperparam

        ## Split Data - Train and Test
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(self.X, 
                                                                                 self.y, 
                                                                                 test_size=self.testsize, 
                                                                                 random_state=55,
                                                                                 shuffle=False)

    ## Compute Prediction and Probability
    def fit_model(self):

        ## Intent Model 
        model = SVC( probability=True, # Allow probability estimate
                     kernel= self.hyperparam['kernel'],  
                     C=self.hyperparam['C'] , # Regularization parameter. A larger value of C implies less regularization.
                     class_weight = cwts(self.y), # Balance Class
                    )

        ## Trai/Fit Model
        model.fit(self.X_train, self.y_train)

        ## Predic Model - Train
        ypred_train = model.predict(self.X_train)
        
        ## Predic Model - Test
        ypred = model.predict(self.X_test)

        ## Probability Model
        yprob = model.predict_proba(self.X_test)[:,1]
        

        return ypred_train, ypred, yprob



#__________________________________________________________________________________
    
#### Class: Blending Ensemble Model 

class BlendEnsemble(ModelEval):

    def __init__(self, X, y, basemodels, metamodel, testsize, valsize):

        ## Call Superclass - initialize attributes from the parent class
        super().__init__(X, y, basemodels, testsize)

        ## Instantiate Meta model     
        self.metamodel = metamodel

        ## Instantiate Validation Size
        self.valsize = valsize
    
        ## Split Data - Train Full and Test 
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.X, 
                                                                                          self.y, 
                                                                                          test_size=self.testsize, 
                                                                                          random_state=55,
                                                                                          shuffle=False )

        ## Split Data - Train and Validation
        self.X_train, self.X_val, self.y_train, self.y_val =  train_test_split( self.X_train_full, 
                                                                                self.y_train_full,
                                                                                test_size=self.valsize, 
                                                                                random_state=55,
                                                                                shuffle=False)

    ## Compute Prediction and Probability
    def fit_model(self):

        ## List to Store Base Models Predictions 
        yhat_val = list()

        for name, model in tqdm(self.models):

            ## Step 1: Trai/Fit Base Models using Train Data (Feature and Target)
            model.fit(self.X_train, self.y_train)

            ## Step 2: Predict Base Models using Validation Feature Data
            yhat = model.predict_proba(self.X_val)[:,1]

            ## Store Predictions as Input for Blending
            yhat_val.append(yhat)

        ## Stack Base Models Predictions 
        yhat_val = np.column_stack(yhat_val)

        ## Instantiate Blending model
        meta_model = self.metamodel

        ## Step 3: Trai/Fit Meta Model
        ##  Use: Base Models Prediction with Validation Feature Data + Validation Target Data 

        meta_model.fit(yhat_val, self.y_val)

        ## List to Store Meta Model Predictions
        yhat_test = list()

        for name, model in self.models:

            ## Step 4(a): Predict Base Models using Test Data
            yhat = model.predict_proba(self.X_test)[:,1]

            ## Append Predictions 
            yhat_test.append(yhat)

        ## Stack Base Models Predictions using Test Data 
        yhat_test = np.column_stack(yhat_test)

        ## Step 4(b): Meta Model Prediction and Probability using Base Model Prediction with Test Feature Data
       ## Predic Model - Test
        ypred_train = meta_model.predict(yhat_val)

       ## Predic Model - Test
        ypred = meta_model.predict(yhat_test)

        ## Probability Model
        yprob = meta_model.predict_proba(yhat_test)[:,1]

        return ypred_train, ypred, yprob
    

        

        #__________________________________________________________________________________
    
