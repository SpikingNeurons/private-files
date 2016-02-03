# -*- coding: utf-8 -*-
"""@package scaffold.crypto.cryptodataprocessor

CryptoDataProcessor class definition
"""

import logging
#from IPython.core.debugger import Tracer

logger = logging.getLogger(__name__)

class CryptoDataProcessor(object):
    """ Class to process requests for CryptoData objects;
        this is the generic class, which is to be extended
        by algorithm-specific subclasses!
    """

    def __init__(self):
        """ __init__ raises an exception if it is not re-implemented
            by a subclass; the algorithms are different enough to require
            a custom __init__ for each specific CryptoDataProcessor!
        """

        raise NotImplementedError("CryptoDataProcessor subclass must"
            + " implement its own innit function!")

    def __check_target(self, target):
        """ Check wether a given CryptoTarget object is valid for the
            CryptoDataProcessor.
        """

        raise NotImplementedError("__check_target must be implemented "
                                  "by a subclass!")

    def __normalize_datum(self, datum):
        """ Normalize a string 'datum' into a tuple that specifies which
            intermediate value was requested.
        """

        raise NotImplementedError("CryptoDataProcessor subclass must"
            + " implement its own __normalize_datum function!")

    def provides(self, datum):
        """ Check wether this CryptoDataProcessor is able to provide the given
            cryptographic datum (i.e. intermediate value).
            datum must be of a specific format, specified in the subclass!
        """

        raise NotImplementedError("Generic CryptoDataProcessor does not"
            + " implement 'provides' function!")


    def _fetch_input(self):
        """ Fetch required input for the specified datum.
            This is very algorithm-specific and must be implemented in
            a subclass!
        """

        raise NotImplementedError("Generic CryptoDataProcessor does not"
            + "implement _fetch_input!")

    def _compute(self):
        """ Compute the requested datum.
            Since computation of a CryptoDatum is unique to each
            algorithm, a subclass must implement this!
        """

        raise NotImplementedError("Generic CryptoDataProcessor does not"
            + " implement _compute!")

    def __getitem__(self, key):
        """ Returns a CryptoDatum object according to `key';
            If the CryptoDatum has not yet been computed,
            it will be computed when this function is trying
            to retrieve it.
            If an invalid datum is requested, an exception is raised.
        """

        raise NotImplementedError("Generic CryptoDataProcessor class does"
            + " not implement __getitem__()!")

    def get_crypto_target(self, datum):
        """ Return a CryptoTarget object that corresponds to the datum.
        """

        raise NotImplementedError("CPD subclasses must implement this!")

    @property
    def crypto_targets(self):
        """ Return a list of all the valid fields that can be requested of
            this processor.
            As this is specific to each algorithm, a subclass must implement this!
        """

        raise NotImplementedError("Generic CryptoDataProcessor class does"
            + " not implement list_fields()!")

    def clear(self):
        """ Clear all previously computed data from the processor.
        """

        self._data = {}

    def guess_data(self, target, guess, trace_range, data_range=None):
        """ Calculate intermediate(s) based on a guessed key rather than
            the key saved in our trace file.
        """

        raise NotImplementedError("guess_data must be implemented "
                                  "by a subclass!")

    @property
    def data(self):
        """ Simple getter for our data.
        """

        return self._data
    

class NoDataProcessor(CryptoDataProcessor):
    """ Empty CryptoDataProcessor that holds no data.
        Kept for backwards compatibility.
    """
    algo_name = 'None'

    def __init__(self, *args, **kwargs):
        logger.warning("Using empty CryptoDataProcessor which holds no data"
            + " and provides no functionality!")
        self._data = None

    def provides(self, *args, **kwargs):
        logger.warning("Using empty CryptoDataProcessor which holds no data"
            + " and provides no functionality!")
        return False
