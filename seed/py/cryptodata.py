# -*- coding: utf-8 -*-
"""@package scaffold.crypto.cryptodata

CryptoData class definition
"""

import logging
import numpy as np
#from IPython.core.debugger import Tracer


logger = logging.getLogger(__name__)

class IntDataCalcError(Exception):
    pass

class CryptoData():
    """Parent class"""
    max_range = None
    num_val = None
    str = None
    label = ''
    round = None
    
    def calc(self, crypto_data, guess, trace_range, data_range):
        raise NotImplementedError
    
    def calc_from_value(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _fetch(crypto_data, field, trace_range, data_range):
        from scaffold.core import TraceFile
        try:
            if isinstance(crypto_data, TraceFile):
                shape = crypto_data.get_field_shape(field)
                return (crypto_data[field, trace_range, data_range]
                        if len(shape) > 1
                        else crypto_data[field][data_range])
            elif isinstance(crypto_data, np.ndarray):
                return crypto_data[field][trace_range, data_range]
            else:
                raise TypeError('expected ndarray or TraceFile, got {}'
                                .format(type(crypto_data)))
        except KeyError:
            raise IntDataCalcError('required data not available')

class NoCryptoData(CryptoData):
    '''
    Null CryptoData class
    '''
    max_range = 0
    num_val = 0


class CryptoDataFromField(CryptoData):

    def __init__(self, field, max_range, num_val):
        self.max_range = max_range
        self.num_val = num_val
        self.str = field
        self.known_input = field
    
    def calc_from_value(self, input, key):
        return input