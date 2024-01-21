"""
# Auxiliary Functions

"""

## Data Manipulation
import numpy as np
import pandas as pd

## Seed
import random # functions for generating random numbers
import tensorflow as tf

## Model Evaluation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Visualization
import matplotlib.pyplot as plt
import os

#--------------------------------------------------


## Set seed
def set_seeds(seed=10): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    


#--------------------------------------------------

### Functions: Evaluation

## Accuracy score train      
def get_accuracy_train(ytrain, ypred):
    score = accuracy_score(ytrain, ypred)
    print(f'Classifier Accuracy Train: {score*100:.4}')


## Accuracy score test
def get_accuracy_test(ytest, ypred):
    score = accuracy_score(ytest, ypred)
    print(f'Classifier Accuracy Test: {score*100:.4}')


## Classification Report
def get_classification_report(ytest, ypred):
    print("Classification Report:")
    print(classification_report(ytest, ypred))

## Confusion Matrix and ROC
def plot_findings(ytest, ypred, yprob, savename):
    
    ## Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ## Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(ytest, ypred, cmap=plt.cm.Reds, ax=ax1)
    ax1.set_title("Confusion Matrix")

    ## Plot ROC Curve
    RocCurveDisplay.from_predictions(ytest, yprob, ax=ax2)
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