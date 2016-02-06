# -*- coding: utf-8 -*-
"""

"""

from cryptodatum import CryptoDatum, CryptoTarget
from cryptodataprocessor import CryptoDataProcessor, NoDataProcessor

import logging
logger = logging.getLogger(__name__)


class SEEDCryptoTarget(CryptoTarget):
    def __init__(self, rnd, step):
        self.rnd = rnd
        self.step = step

    def __str__(self):
        return str(self.rnd) + ':' + str(self.step)

    def __hash__(self):
        pass


class SEEDCryptoDatum(CryptoDatum):
    def __init__(self, inpt, rnd, step):
        self.rnd = rnd
        self.step = step
        self.datum = inpt # processed input will get stored here

    def __str__(self):
        return str(self.rnd) + ':' + str(self.step)


class SEEDCryptoDataProcessor(CryptoDataProcessor):

    def _fetch_input(self):
        pass

    def list_fields(self):
        pass

    def __init__(self, tf, **kwargs):
        self.trace_file = tf
        self._data = {}

    def provides(self, query):
        if query[-4:] == 'SEED':
            return True
        return False

    def __getitem__(self, key):
        target = key[0]
        self._compute()
        return self.data

    def get_crypto_target(self, query):
        rnd, step = query.split(':')
        rnd = int(rnd)
        return SEEDCryptoTarget(rnd, step)

    def _compute(self):
        self.data = self.trace_file['plaintext']
        self.data[0, 15] += 1

