# -*- coding: utf-8 -*-
"""@package scaffold.crypto.cryptodatum

CryptoDatum class definition
"""

import numpy as np

from scaffold.core import DataTarget

# import and instantiate logging
import logging
logger = logging.getLogger(__name__)

class DataNotFoundError(Exception):
    """ Exception to be thrown if a piece of data requested from a
        TraceFile object is not available.
    """
    pass

class CryptoTarget(DataTarget):
    """ Generic class that holds meta-information on cryptographic
        intermediates, but no actual cryptographic data.
    """

    def __init__(self):
        """ Initialization is algorithm-dependant, so is not
            implemented generically.
        """

        raise NotImplementedError("__init__ must be implemented individually "
                                  "for every algorithm!")
    def __str__(self):
        """ Mandatory, but algorithm-dependant function.
        """

        raise NotImplementedError("__str__ must be implemented individually "
                                  "for every algorithm!")

    def __hash__(self):
        """ CryptoTarget can be hashed, and thus used as dictionary-index!
        """

        raise NotImplementedError("__hash__ must be implemented "
                                  "individually and safely for every algorithm!")

class CryptoDatum(object):
    """ Generic class to hold intermediate value and meta-data for 
        a cryptographic algorithm.
        This class is intended as a blueprint to inherit from,
        not to be used itself!
    """

    def __init__(self):
        """ init function depends heavily on the specific
            algorithm!
        """

        logger.warning("Received call to generic class which should not be used"
            + " explicitly! Raising exception...")

        raise NotImplementedError("Init function must be implemented"
            + " by algorithm-specific class!")

    @property
    def datum(self):
        """ Simple getter method to fetch the datum the respective object saves
        """
        return self._datum

    @staticmethod
    def fetch(trace_file, field, trace_range=None, data_range=None):
        """ Method to fetch information from a TraceFile object.
            Parameters:
            trace_file : TraceFile
                trace file to search
            field : str
                the field that is requested, one of:
                plaintext, ciphertext, key
            trace_range : slice
                the range of traces to consider
            data_range : slice
                the column to consider
        """

        ret = trace_file[field]
        if field == 'plaintext':
            if data_range is not None and trace_range is not None:
                return ret[trace_range, data_range]
            elif data_range is None and trace_range is not None:
                return ret[trace_range, slice(None)]
            elif data_range is not None and trace_range is None:
                return ret[slice(None), data_range]
            else:
                return ret
        elif field == 'key' or field == 'ciphertext':
            return ret
        else:
            raise ValueError("Unable to fetch '{}' field!".format(field))

        # try to fetch the information from the tracefile
        # try:
        #     shape = trace_file.get_field_shape(field)
        #     if trace_range is None and data_range is None:
        #         return trace_file[field]
        #     elif trace_range is None:
        #         return trace_file[field][data_range]
        #     # elif data_range is None:
        #     #     return trace_file[field, trace_range]
        #     else:
        #         return (trace_file[field, trace_range, data_range]
        #                 if len(shape) > 1
        #                 else trace_file[field][data_range])

        # # intercept a KeyError and raise a more appropriate error
        # except KeyError:
        #     raise DataNotFoundError('Requested data not available!')
