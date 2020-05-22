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
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


class SingleLayerNN(pints.NNEmulator):
    """
    Single layer neural network emulator.
    Extends :class:`NNEmulator`.
    """
    def __init__(self, log_likelihood, X, y, input_scaler=False, output_scaler=False):
        super(SingleLayerNN, self).__init__(log_likelihood, X, y, input_scaler=False, output_scaler=False)
        self._model = Sequential()
        #Sequential([
          #Dense(64, activation='relu'),
          #Dense(64, activation='relu'),
          #Dense(10, activation='softmax'),
        #])

    def __call__(self, x):
        """ Additional **kwargs can be provided to Keras's predict method. """
        #if not isinstance(x, list):
        x = np.array(x).reshape((1, self.n_parameters()))
        if self._input_scaler:
            x = self._input_scaler.transform(x)

        y = self._model.predict([x])
        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)
        return y

    def set_parameters(self, neurons=64, activation='linear', optimizer='adam', 
                       loss='mean_squared_error', metrics=['accuracy']):
        """ Provide parameters to compile the model. """
        self._model.add(Dense(neurons,
                                activation=tf.nn.leaky_relu,
                                input_shape=(super().n_parameters(),),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01),)
        )
        self._model.add(Dense(1, activation=activation))  # output layer
        
        opt = SGD(lr=0.001)
        self._model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics
        )

    def fit(self, epochs=50, batch_size=32, validation_split=0.2):
        """ Training neural network and return history. """
        self._history = self._model.fit(self._X,
                                    self._y,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    shuffle=True
        )
        return self._history

    def summary(self):
        return self._model.summary()

    def evaluate(self, x, **kwargs):
        """ Uses Keras's evaluate() method, so can provide additional paramaters. """
        return self._model.evaluate(x, **kwargs)

    def get_model(self):
        """ Return model. """
        return self._model

    def get_model_history(self):
        """ Returns the log marginal likelihood of the model. """
        assert hasattr(self, "_history"), "Must first train NN"
        return self._history

    def name(self):
        """ See :meth:`pints.NNEmulator.name()`. """
        return 'Single layer neural network'
    
    
class MultiLayerNN(pints.NNEmulator):
    """
    Single layer neural network emulator.
    Extends :class:`NNEmulator`.
    """
    def __init__(self, log_likelihood, X, y, input_scaler=False, output_scaler=False):
        super(MultiLayerNN, self).__init__(log_likelihood, X, y, input_scaler=False, output_scaler=False)
        self._model = Sequential()
        #Sequential([
          #Dense(64, activation='relu'),
          #Dense(64, activation='relu'),
          #Dense(10, activation='softmax'),
        #])

    def __call__(self, x):
        """ Additional **kwargs can be provided to Keras's predict method. """
        #if not isinstance(x, list):
        x = np.array(x).reshape((1, self.n_parameters()))
        if self._input_scaler:
            x = self._input_scaler.transform(x)

        y = self._model.predict([x])
        if self._output_scaler:
            y = self._output_scaler.inverse_transform(y)
        return y

    def set_parameters(self, neurons=64, activation='linear', optimizer='adam', 
                       loss='mean_squared_error', metrics=['accuracy']):
        """ Provide parameters to compile the model. """
        self._model.add(Dense(64,
            activation=tf.nn.leaky_relu,
            input_shape=(super().n_parameters(),),
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))
        self._model.add(Dense(128,
            activation=tf.nn.leaky_relu,
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))
        self._model.add(Dense(256,
            activation=tf.nn.leaky_relu,
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))
        self._model.add(Dense(128,
            activation=tf.nn.leaky_relu,
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))
        self._model.add(Dense(64,
            activation=tf.nn.leaky_relu,
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))
        self._model.add(Dense(1, activation=activation))  # output layer
        self._model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )

    def fit(self, epochs=50, batch_size=32, validation_split=0.2):
        """ Training neural network and return history. """
        print(self._X)
        print(self._y)
        self._history = self._model.fit(self._X,
                                    self._y,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    shuffle=True
        )
        return self._history

    def summary(self):
        return self._model.summary()

    def evaluate(self, x, **kwargs):
        """ Uses Keras's evaluate() method, so can provide additional paramaters. """
        return self._model.evaluate(x, **kwargs)

    def get_model(self):
        """ Return model. """
        return self._model

    def get_model_history(self):
        """ Returns the log marginal likelihood of the model. """
        assert hasattr(self, "_history"), "Must first train NN"
        return self._history

    def name(self):
        """ See :meth:`pints.NNEmulator.name()`. """
        return 'Single layer neural network'