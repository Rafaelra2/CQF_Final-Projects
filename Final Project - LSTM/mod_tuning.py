"""
# Mod Tuning

This code tune 3 models

"""

## Data Manipulation
import numpy as np
import pandas as pd

## Visualization
import matplotlib.pyplot as plt

## Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall

## Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


#__________________________________________________________________________________


## Import Variable
lookback  = pd.read_csv('../Output/Variables/seqlen.csv', header=None).iloc[0, 0]
features  = pd.read_csv('../Output/Variables/numfeat.csv', header=None).iloc[0, 0]

   
#__________________________________________________________________________________

## My Callback

def tune_callbacks( monitor='loss', patience=10):
                                                          
    callbacks = [ EarlyStopping( patience= patience, # 5-20 Training will continue for x epochs even after it starts detecting no improvement
                                 monitor=monitor, # Set the Loss function
                                 mode='min', # Training will stop when the quantity monitored has stopped decreasing.
                                 verbose=0, # Detailed output
                                 restore_best_weights=True) # Ensures that the  weights are reverted to those of the epoch with the best value 
                    ]
    return callbacks

#__________________________________________________________________________________




#__________________________________________________________________________________

### Tune = 1 Hidden Layers

def tune_lstm_h1(hp):
        
    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()
    
    ## Tune the number of units in the layers
    hp_units1 = hp.Int('un1', min_value=4, max_value=24, step=4)
  
    ## Tune activation functions
    hp_activation1 = hp.Choice(name = 'activ1', values = ['relu', 'elu'], ordered = False)

    ## Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    
    ## Add an LSTM Layer
    model.add( LSTM( hp_units1, 
                     input_shape=(lookback, features), 
                     activation=hp_activation1, 
                     return_sequences=False, 
                     name='lstm_1')
               )   

    ## Add a Dense Layer
    model.add( Dense( units=1, 
                     activation='sigmoid', 
                     name='output')
              )    

   ## Set Optimizater
    opt = Adam( lr=hp_learning_rate, 
                epsilon=1e-08, 
                decay=0.0
               )       
    
    ## Model compilation 
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy',  
                           Precision(),
                           Recall()]
                  )

    return model


 #____________________________________________________________________________
 
 
 
 
 
#__________________________________________________________________________________

### Tune = 2 Hidden Layers

def tune_lstm_h2(hp):
        
    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()
    
    ## Tune the number of units in the layers
    hp_units1 = hp.Int('un1', min_value=4, max_value=24, step=4)
    hp_units2 = hp.Int('un2', min_value=4, max_value=24, step=4)

    ## Tune activation functions
    hp_activation1 = hp.Choice(name = 'activ1', values = ['relu', 'elu'], ordered = False)
    hp_activation2 = hp.Choice(name = 'activ2', values = ['relu', 'elu'], ordered = False)

    ## Tune the dropout rate
    hp_dropout1 = hp.Float('dropout1', min_value=0, max_value=0.5, step=0.1)

    ## Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    
    ## Add an LSTM Layer
    model.add( LSTM( hp_units1, 
                     input_shape=(lookback, features), 
                     activation=hp_activation1, 
                     return_sequences=True, 
                     name='lstm_1')
               )   
    ## Add Dropout
    model.add( Dropout(hp_dropout1, name='drouput1') )

    ## Add an LSTM Layer
    model.add( LSTM(hp_units2, 
                    activation = hp_activation2, 
                    return_sequences=False, 
                    name='lstm_2')
              )

    ## Add a Dense Layer
    model.add( Dense( units=1, 
                     activation='sigmoid', 
                     name='output')
              )    

   ## Set Optimizater
    opt = Adam( lr=hp_learning_rate, 
                epsilon=1e-08, 
                decay=0.0
               )       
    
    ## Model compilation 
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy',  
                           Precision(),
                           Recall()]
                  )

    return model


 #____________________________________________________________________________






#__________________________________________________________________________________

### Tune = 3 Hidden Layers

def tune_lstm_h3(hp):
        
    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()
    
    ## Tune the number of units in the layers
    hp_units1 = hp.Int('un1', min_value=4, max_value=24, step=4)
    hp_units2 = hp.Int('un2', min_value=4, max_value=24, step=4)
    hp_units3 = hp.Int('un3', min_value=4, max_value=24, step=4)
  
    ## Tune activation functions
    hp_activation1 = hp.Choice(name = 'activ1', values = ['relu', 'elu'], ordered = False)
    hp_activation2 = hp.Choice(name = 'activ2', values = ['relu', 'elu'], ordered = False)
    hp_activation3 = hp.Choice(name = 'activ3', values = ['relu', 'elu'], ordered = False)

    ## Tune the dropout rate
    hp_dropout1 = hp.Float('dropout1', min_value=0, max_value=0.5, step=0.1)
    hp_dropout2 = hp.Float('dropout2', min_value=0, max_value=0.5, step=0.1)

    ## Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    
    ## Add an LSTM Layer
    model.add( LSTM( hp_units1, 
                     input_shape=(lookback, features), 
                     activation=hp_activation1, 
                     return_sequences=True, 
                     name='lstm_1')
               )   
    ## Add Dropout
    model.add( Dropout(hp_dropout1, name='drouput1') )

    ## Add an LSTM Layer
    model.add( LSTM(hp_units2, 
                    activation = hp_activation2, 
                    return_sequences=True, 
                    name='lstm_2')
              )
    ## Add Dropout
    model.add( Dropout(hp_dropout2, 
                       name='drouput2')
               )

    ## Add an LSTM Layer
    model.add( LSTM( hp_units3, 
                     activation = hp_activation3,
                       return_sequences=False, 
                       name='lstm_3')
              )

    ## Add a Dense Layer
    model.add( Dense( units=1, 
                     activation='sigmoid', 
                     name='output')
              )    

   ## Set Optimizater
    opt = Adam( lr=hp_learning_rate, 
                epsilon=1e-08, 
                decay=0.0
               )       
    
    ## Model compilation 
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy',  
                           Precision(),
                           Recall()]
                  )

    return model


 #__________________________________________________________________________________ 
  
  
    
 




    
