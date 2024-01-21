"""
# Base Functions

"""

## Data Manipulation
import numpy as np
import pandas as pd

## Seeds
import random # functions for generating random numbers
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
#________________________________________________________________________

from sklearn.base import BaseEstimator, TransformerMixin

# create custom day transformer 
from sklearn.base import BaseEstimator, TransformerMixin

# create custom day transformer 
class DayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Mapping of weekday names to numeric values (Monday=1, Sunday=7)
        self.weekday_map = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 
            'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }

    def fit(self, X, y=None):
        return self  # Nothing to do in fit method for this transformer

    def transform(self, X):
        # Ensure X is a DataFrame
        if isinstance(X, pd.Series):
            Xt = X.to_frame()
        else:
            Xt = X.copy()

        # Convert weekday names to numbers
        Xt['weekday_num'] = Xt.iloc[:, 0].map(self.weekday_map)

        # Calculate sine and cosine transformations
        pi = np.pi
        Xt['dsin'] = np.sin(2 * pi * Xt['weekday_num'] / 7)
        Xt['dcos'] = np.cos(2 * pi * Xt['weekday_num'] / 7)


        # Drop the original days column if it exists
        Xt = Xt.drop(Xt.columns[0], axis=1)
        Xt = Xt.drop('weekday_num', axis=1)

        return Xt


## Class weight function - General
def cwts(dfs):
    c0, c1 = np.bincount(dfs)
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}


## Class weight function - XGBoost
def cwts_scale(dfs):
    c0, c1 = np.bincount(dfs)
    scale_pos_weight = c0 / c1
    return scale_pos_weight

## Seed
def set_seeds(seed=55): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)