# -*- coding: utf-8 -*-

from collections import namedtuple

# seed algo meta info
SEED_ALGO = namedtuple('SEED_ALGO',
                       'bits_supported, max_rounds, header, footer')
SEED_ALGO.bits_supported = [128]
SEED_ALGO.max_rounds = 16
SEED_ALGO.header = '\n\n>>> SEED Algorithm: '
SEED_ALGO.footer = '\n\n'

# seed step info
STEPS_PROVIDED = \
    namedtuple('STEPS_PROVIDED',
               'RoundKey, Right, AddRoundKey, GDa, GC, GDb, F, Output')
STEP_STRUCT = namedtuple('STEP_STRUCT',
                         'id, num_of_bytes')
STEPS_PROVIDED.RoundKey    = STEP_STRUCT(id=0, num_of_bytes=8)
STEPS_PROVIDED.Right       = STEP_STRUCT(id=1, num_of_bytes=8)
STEPS_PROVIDED.AddRoundKey = STEP_STRUCT(id=2, num_of_bytes=8)
STEPS_PROVIDED.GDa         = STEP_STRUCT(id=3, num_of_bytes=4)
STEPS_PROVIDED.GC          = STEP_STRUCT(id=4, num_of_bytes=4)
STEPS_PROVIDED.GDb         = STEP_STRUCT(id=5, num_of_bytes=4)
STEPS_PROVIDED.F           = STEP_STRUCT(id=6, num_of_bytes=8)
STEPS_PROVIDED.Output      = STEP_STRUCT(id=7, num_of_bytes=16)


class SEEDCryptoTarget:
    """
    Object interface for target query.
    It does not hold any computed information.
    """

    def __init__(self, trace_file, rnd, step, keysize, decrypt):
        """
        Constructor for SEEDCryptoTarget

        Parameters
        ----------
        trace_file : TraceFile
            Input trace file which hold meta info plus data.
        rnd : int
            round you want to check (1 to 16)
        step : int
            step you want to check
        keysize : int
            the size of the key (it is fixed to 128-bit for SEED)
        decrypt : bool
            select decrypt or encrypt

        Returns
        -------
        SEEDCryptoTarget
            SEEDCryptoTarget object that can be used with TraceFileHandler

        """

        self.rnd = rnd
        self.step = step

        if keysize is not 128:
            raise ValueError(HEADER + 'Keysize can be only 128 bit' + FOOTER)

        self.keysize = keysize
        self.decrypt = decrypt

        # Determine num_val, the number of values a single element of the
        # intermediate might take. This is constant in case of SEED as every
        #  element of intermediate step is made up of 8 bits.
        num_val = 2**8

        # max_range, or the number of elements in each intermediate-array.
        # This is not constant and can take different value based on step
        # selected.
        for step_ in STEPS_PROVIDED_AND_MAX_RANGE:
            if step_[0] == step:
                max_range = step_[2]
                break
        else:
            raise ValueError(
                HEADER + "Invalid step: " + str(step) +
                "\n You are allowed to use following steps: " +
                [x[0] for x in STEPS_PROVIDED_AND_MAX_RANGE] + FOOTER)

        # based on step selected check if the field is local
        if self.step == 'RoundKey':
            local = trace_file.has_local_field('key')
        elif self.decrypt:
            local = trace_file.has_local_field('ciphertext')
        else:
            local = trace_file.has_local_field('plaintext')

        # set next and previous target to None;
        # will be computed as needed.
        self._next = None
        self._previous = None

        # init superclasses
        super().__init__(trace_file, max_range, num_val, local)





















