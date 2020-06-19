#
# Basic emulator artificial neural network (ANN)
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import HDF5Matrix


class NerualNetwork(pints.ProblemLogLikelihood):
    """
    Abstract base class for emulators that are based on neural networks.
    Extends :class:`Emulator`.
    Parameters
    ----------
    x0
        An starting point in the parameter space.
    sigma0
        An optional (initial) covariance matrix, i.e., a guess of the
        covariance of the distribution to estimate, around ``x0``.
    """
    
    """
    *Extends:* :class:`LogPDF`
    Abstract class from which all emulators should inherit.
    An instance of the Emulator models given log-likelihood
    Arguments:
    ``log_likelihood``
        A :class:`LogPDF`, the likelihood distribution being emulated.
    ``X``
        N by n_paremeters matrix containing inputs for training data
    ``y``
        N by 1, target values for each input vector
    ``input_scaler``
        sklearn scalar type, don't pass class just the type.
        E.g. StandardScaler provides standardization.
    ``output_scaler``
        sklearn scaler class that will be applied to output
    """

    def __init__(self, problem, X, y, input_scaler=None, output_scaler=None):
        # Perform sanity checks for given data
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError("Given problem must extend LogPDF")
        super(NerualNetwork, self).__init__(problem)

        # Store counts
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()

        # check if dimensions are valid
        if X.ndim != 2:
            raise ValueError("Input should be 2 dimensional")
        X_r, X_c = X.shape
        if (X_c != self._n_parameters):
            raise ValueError("Input data should have n_parameters features")

        # if given target array is 1d convert automatically
        if y.ndim == 1:
            y = y.reshape(len(y), 1)
        if y.ndim != 2:
            raise ValueError("Target array should be 2 dimensional (N, 1)")

        y_r, y_c = y.shape
        if y_c != 1:
            raise ValueError("Target array should only have 1 feature")
        if (X_r != y_r):
            raise ValueError("Input and target dimensions don't match")

        # Scale data for inputs and output
        self._input_scaler = input_scaler
        if input_scaler:
            self._input_scaler.fit(X)
            self._X = self._input_scaler.transform(X)
        else:
            self._X = copy.deepcopy(X)  # use a copy to prevent original data from changing

        self._output_scaler = output_scaler
        if output_scaler:
            self._output_scaler.fit(y)
            self._y = self._output_scaler.transform(y)
        else:
            self._y = copy.deepcopy(y)  # use a copy to prevent original data from changing

    def n_parameters(self):
        return self._n_parameters
    

class SingleLayerNN(NeuralNetwork):
    """
    Single layer neural network emulator.
    Extends :class:`NNEmulator`.
    """
    def __init__(self, log_likelihood, X, y, input_scaler=None, output_scaler=None):
        super(SingleLayerNN, self).__init__(log_likelihood, X, y, input_scaler, output_scaler)
        self._model = Sequential()


    def __call__(self, x):
        """ Additional **kwargs can be provided to Keras's predict method. """
        x = np.array(x).reshape((1, self.n_parameters()))
        if self._input_scaler:
            x = self._input_scaler.transform(x)

        y = self._model.predict([x])
        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)
        return y

    def set_parameters(self, neurons=64, hidden_activation='relu', activation='sigmoid', 
                       learning_rate=0.001, loss='mse', metrics=['mae']):
        """ Provide parameters to compile the model. """
        self._model.add(Dense(neurons,
                                activation=hidden_activation,
                                input_dim=2,
                                kernel_initializer='he_uniform'
                                #kernel_regularizer=tf.keras.regularizers.l2(0.01),)
        ))
        self._model.add(Dense(1, activation=activation))  # output layer
        
        opt = keras.optimizers.SGD(learning_rate, momentum=0.9)
        #opt = keras.optimizers.Adam(learning_rate)
        self._model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics
        )

    def fit(self, epochs=50, batch_size=32, X_val=None, y_val=None, verbose=0):
        """ Training neural network and return history. """
        if self._input_scaler and X_val is not None:
            X_val = self._input_scaler.transform(X_val)
        if self._output_scaler and y_val is not None:
            y_val = np.array(y_val).reshape((len(y_val),1))
            y_val = self._output_scaler.transform(y_val)
                                            
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20)
        self._history = self._model.fit(self._X,
                                        self._y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks=[rlrop],
                                        shuffle=True,
                                        validation_data=(X_val, y_val),
                                        verbose=verbose
        )
        return self._history

    def summary(self):
        return self._model.summary()

    def evaluate(self, X_test, y_test, **kwargs):
        """ Uses Keras's evaluate() method, so can provide additional paramaters. """
        if self._input_scaler:
            X_test = self._input_scaler.transform(X_test)
        if self._output_scaler:
            y_test = self._output_scaler.inverse_transform(y_test)  
        return self._model.evaluate(X_test, y_test, **kwargs)

    def get_model(self):
        """ Return model. """
        return self._model

    def get_model_history(self):
        """ Returns the log marginal likelihood of the model. """
        assert hasattr(self, "_history"), "Must first train NN"
        return self._history

    def name(self):
        """ See :meth:`pints.NNEmulator.name()`. """
        return 'Single-layer neural network'
    
    
class MultiLayerNN(NeuralNetwork):
    """
    Single layer neural network emulator.
    Extends :class:`NNEmulator`.
    """
    def __init__(self, log_likelihood, X, y, input_scaler=None, output_scaler=None):
        super(MultiLayerNN, self).__init__(log_likelihood, X, y, input_scaler, output_scaler)
        self._model = Sequential()


    def __call__(self, x):
        """ Additional **kwargs can be provided to Keras's predict method. """
        x = np.array(x).reshape((1, self.n_parameters()))
        if self._input_scaler:
            x = self._input_scaler.transform(x)

        y = self._model.predict([x])
        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)
        return y

    def set_parameters(self, layers=3, neurons=64, hidden_activation='relu', activation='sigmoid', 
                       learning_rate=0.001, loss='mse', metrics=['mae']):
        """ Provide parameters to compile the model. """
        # Input layer
        self._model.add(Dense(neurons,
                                activation=hidden_activation,
                                input_dim=2,
                                kernel_initializer='he_uniform'
                                #kernel_regularizer=tf.keras.regularizers.l2(0.01),)
        ))
        # Hidden layers
        for n in range(2, layers+1):    
            self._model.add(Dense(n*neurons,
                activation=tf.nn.leaky_relu,
                kernel_initializer='he_uniform'                  
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ))
        for n in range(layers, 1, -1):    
            self._model.add(Dense(n*neurons,
                activation=tf.nn.leaky_relu,
                kernel_initializer='he_uniform'                  
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ))
        # Output layer
        self._model.add(Dense(1, activation=activation))
        
        opt = keras.optimizers.SGD(learning_rate, momentum=0.9)
        #opt = keras.optimizers.Adam(learning_rate)
        self._model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics
        )

    def fit(self, epochs=50, batch_size=32, X_val=None, y_val=None, verbose=0):
        """ Training neural network and return history. """
        if self._input_scaler and X_val is not None:
            X_val = self._input_scaler.transform(X_val)
        if self._output_scaler and y_val is not None:
            y_val = np.array(y_val).reshape((len(y_val),1))
            y_val = self._output_scaler.transform(y_val)  
            
        early_stopper = EarlyStopping(monitor='val_loss',min_delta=0,patience=50,verbose=0,mode='auto')
        plateau_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=50)
        self._history = self._model.fit(self._X,
                                        self._y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks=[early_stopper],
                                        shuffle=True,
                                        validation_data=(X_val, y_val),
                                        verbose=verbose
        )
        return self._history

    def summary(self):
        return self._model.summary()

    def evaluate(self, X_test, y_test, **kwargs):
        """ Uses Keras's evaluate() method, so can provide additional paramaters. """
        if self._input_scaler:
            X_test = self._input_scaler.transform(X_test)
        if self._output_scaler:
            y_test = self._output_scaler.inverse_transform(y_test)  
        return self._model.evaluate(X_test, y_test, **kwargs)

    def get_model(self):
        """ Return model. """
        return self._model

    def get_model_history(self):
        """ Returns the log marginal likelihood of the model. """
        assert hasattr(self, "_history"), "Must first train NN"
        return self._history

    def name(self):
        """ See :meth:`pints.NNEmulator.name()`. """
        return 'Multi-layer neural network'

