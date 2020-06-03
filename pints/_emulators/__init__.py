#
# Sub-module containing likelihood emulators
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import numpy as np
import copy


class Emulator(pints.Loggable, pints.TunableMethod, pints.LogPDF): #pints.LogPDF
    """
    Abstract base class for likelihood function emulators.
    All emulators implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.
    """

    def name(self):
        """
        Returns this emulator's full name.
        """
        raise NotImplementedError

    def in_initial_phase(self):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method returns ``True`` if the
        method is currently configured to be in its initial phase. For other
        methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def needs_initial_phase(self):
        """
        Returns ``True`` if this method needs an initial phase, for example an
        adaptation-free period for adaptive covariance methods, or a warm-up
        phase for DREAM.
        """
        return False

    def set_initial_phase(self, in_initial_phase):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method toggles the initial phase
        algorithm. For other methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError
    
    
class NNEmulator(Emulator):
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

    def __init__(self, log_likelihood, X, y, input_scaler=None, output_scaler=None):
        # Perform sanity checks for given data
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError("Given pdf must extend LogPDF")
        self._n_parameters = log_likelihood.n_parameters()

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