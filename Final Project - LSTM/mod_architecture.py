"""
# Mod Architecture

1.1) LSTM - 1 Hidden Layer no tuning
2.1) LSTM - 2 Hidden Layers no tuning
3.1) LSTM - 3 Hidden Layers no tuning

1.2) LSTM - 1 Hidden Layer with tuning
2.2) LSTM - 2 Hidden Layers with tuning
3.2) LSTM - 3 Hidden Layers with tuning

"""

### Libraries

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

## Other
import os
import datetime as dt


#__________________________________________________________________________________

## Callback Final Models

def create_callbacks(filepath, log_dir, monitor='loss', patience=10):
                                                          
    callbacks = [ EarlyStopping( patience= patience, # 5-20 Training will continue for x epochs even after it starts detecting no improvement
                                 monitor=monitor, # Set the Loss function
                                 mode='min', # Training will stop when the quantity monitored has stopped decreasing.
                                 verbose=1, # Detailed output
                                 restore_best_weights=True), # Ensures that the  weights are reverted to those of the epoch with the best value 
                 ModelCheckpoint( filepath=filepath,  
                                  verbose=1, 
                                  monitor=monitor, # Set the Loss function
                                  mode='min', # Set the Loss function
                                  save_best_only=True),
                    TensorBoard(log_dir= log_dir, histogram_freq=1 )
                    ]
    return callbacks

#__________________________________________________________________________________




#__________________________________________________________________________________
### LSTM 1.1 = 1 Hidden Layer and no Tuning

def create_lstm_h1_noTun(hu=256, lookback=21, features=1):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=hu, # Set a higher number of initial hidden layers because most of them will be drop after the dropouts
                      input_shape=(lookback, features), 
                      activation = 'elu', # Exponential Linear Unit
                      return_sequences=False, # LSTM layer return for the next layer   
                      name='lstm_1')
                )
   
    ## Add a Dense Layer
    model.add( Dense( units=1, # Layer has a single neuron which will output a single value.
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  # RMSprop (Root Mean Square Propagation) is other good option
    opt = Adam( lr=0.001, # Learning Rate
                epsilon=1e-08, # Threshold Error for optimization
                decay=0.0 # Learning rate decay 
                )       
    
    ## Model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), # Metric used for training. Common choice for classification problems
                   metrics=[ 'accuracy', # Metric used for evaluation purpose
                              Precision(),
                              Recall()]
                  )

    return model
    
#__________________________________________________________________________________



#__________________________________________________________________________________
### LSTM 2.1 = 2 Hidden Layers and no Tuning

def create_lstm_h2_noTun(hu=256, dropout=0.5, lookback=21, features=1):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=hu*2, # Set a higher number of initial hidden layers because most of them will be drop after the dropouts
                      input_shape=(lookback, features), 
                      activation = 'elu', # Exponential Linear Unit
                      return_sequences=True, # LSTM layer return for the next layer   
                      name='lstm_1')
                )
    ## Add Dropbout
    model.add( Dropout(dropout, name='dropout_1') )
   
    ## Add an LSTM Layer
    model.add( LSTM( units=hu, 
                     activation = 'elu', 
                     return_sequences=False, 
                     name='lstm_2')
                )
    
    ## Add a Dense Layer
    model.add( Dense( units=1, # Layer has a single neuron which will output a single value.
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  # RMSprop (Root Mean Square Propagation) is other good option
    opt = Adam( lr=0.001, # Learning Rate
                epsilon=1e-08, # Threshold Error for optimization
                decay=0.0 # Learning rate decay 
                )       
    
    ## Model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), # Metric used for training. Common choice for classification problems
                   metrics=[ 'accuracy', # Metric used for evaluation purpose
                              Precision(),
                              Recall()]
                  )

    return model
#__________________________________________________________________________________




#__________________________________________________________________________________
### LSTM 3.1 = 3 Hidden Layers and no Tuning

def create_lstm_h3_noTun(hu=256, dropout=0.5, lookback=21, features=1):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=hu*3, # Set a higher number of initial hidden layers because most of them will be drop after the dropouts
                      input_shape=(lookback, features), 
                      activation = 'elu', # Exponential Linear Unit
                      return_sequences=True, # LSTM layer return for the next layer   
                      name='lstm_1')
                )
    ## Add Dropbout
    model.add( Dropout(dropout, name='dropout_1') )

    ## Add an LSTM Layer
    model.add( LSTM(units=hu*2, 
                    activation = 'elu',  # Exponential Linear Unit
                    return_sequences=True, # LSTM layer return for the next layer 
                    name='lstm_2')
              )
                    
    ## Add Dropbout
    model.add( Dropout(dropout, name='dropout_2') )
    
    ## Add an LSTM Layer
    model.add( LSTM( units=hu, 
                     activation = 'elu', 
                     return_sequences=False, 
                     name='lstm_3')
                )
    
    ## Add a Dense Layer
    model.add( Dense( units=1, # Layer has a single neuron which will output a single value.
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  # RMSprop (Root Mean Square Propagation) is other good option
    opt = Adam( lr=0.001, # Learning Rate
                epsilon=1e-08, # Threshold Error for optimization
                decay=0.0 # Learning rate decay 
                )       
    
    ## Model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), # Metric used for training. Common choice for classification problems
                   metrics=[ 'accuracy', # Metric used for evaluation purpose
                              Precision(),
                              Recall()]
                  )

    return model
#__________________________________________________________________________________







#__________________________________________________________________________________
### LSTM 1.2 = 1 Hidden Layer + Tuning

def create_lstm_h1_Tun( best, lookback, features ):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=best['un1'], 
                      input_shape=(lookback, features), 
                      activation = best['activ1'],
                      return_sequences=False,
                      name='lstm_1')
                )
                    
    ## Add a Dense Layer
    model.add( Dense( units=1, 
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  
    opt = Adam( lr=best['lr'], 
                epsilon=1e-08, 
                decay=0.0 
                )       
    
    ## Model compilation)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), 
                   metrics=[ 'accuracy', 
                              Precision(),
                              Recall()]
                  )

    return model
#__________________________________________________________________________________









#__________________________________________________________________________________
### LSTM 2.2 = 2 Hidden Layers + Tuning

def create_lstm_h2_Tun( best, lookback, features ):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=best['un1'], 
                      input_shape=(lookback, features), 
                      activation = best['activ1'],
                      return_sequences=True,
                      name='lstm_1')
                )
    ## Add Dropbout
    model.add( Dropout(best['dropout1'], name='dropout_1') )

    ## Add an LSTM Layer
    model.add( LSTM(units=best['un2'], 
                    activation = best['activ2'],  
                    return_sequences=False, 
                    name='lstm_2')
              )
                    

    ## Add a Dense Layer
    model.add( Dense( units=1, 
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  
    opt = Adam( lr=best['lr'], 
                epsilon=1e-08, 
                decay=0.0 
                )       
    
    ## Model compilation)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), 
                   metrics=[ 'accuracy', 
                              Precision(),
                              Recall()]
                  )

    return model
#__________________________________________________________________________________









    

#__________________________________________________________________________________
### LSTM 3.2 = 3 Hidden Layers + Tuning

def create_lstm_h3_Tun( best, lookback, features ):

    ## Resetting the State: 
    tf.keras.backend.clear_session()   

    ## Instantiate the model
    model = Sequential()

    ## Add an LSTM Layer
    model.add(  LSTM( units=best['un1'], 
                      input_shape=(lookback, features), 
                      activation = best['activ1'],
                      return_sequences=True,
                      name='lstm_1')
                )
    ## Add Dropbout
    model.add( Dropout(best['dropout1'], name='dropout_1') )

    ## Add an LSTM Layer
    model.add( LSTM(units=best['un2'], 
                    activation = best['activ2'],  
                    return_sequences=True, 
                    name='lstm_2')
              )
                    
    ## Add Dropbout
    model.add( Dropout(best['dropout2'], name='dropout_2') )
    
    ## Add an LSTM Layer
    model.add( LSTM( units=best['un3'], 
                     activation = best['activ3'], 
                     return_sequences=False, 
                     name='lstm_3')
                )
    
    ## Add a Dense Layer
    model.add( Dense( units=1, 
                      activation='sigmoid', 
                      name='output')
               )             
    
    ## Specify optimizer  
    opt = Adam( lr=best['lr'], 
                epsilon=1e-08, 
                decay=0.0 
                )       
    
    ## Model compilation)
    model.compile( optimizer=opt, 
                   loss=BinaryCrossentropy(), 
                   metrics=[ 'accuracy', 
                              Precision(),
                              Recall()]
                  )

    return model
#__________________________________________________________________________________

