# -*- coding: utf-8 -*-

# ................................................................. some imports
from collections import namedtuple
# .......................................................................... end

# .......................................................... SEED algo meta info
SEED_ALGO = namedtuple('SEED_ALGO',
                       'bits_supported, max_rounds, num_val, header, footer')
# only 128 bits are supported
SEED_ALGO.bits_supported = [128]
# SEED is made up of 16 rounds
SEED_ALGO.max_rounds = 16
# Intermediate value is made up of 8 bits and can take 2^8 values
SEED_ALGO.num_val = 2**8
# header and footer
SEED_ALGO.header = '\n\n>>> SEED Algorithm: '
SEED_ALGO.footer = '\n\n'
# .......................................................................... end

# ............................................................... SEED step info
STEPS_PROVIDED = \
    namedtuple('StepsProvided',
               'RoundKey, Right, AddRoundKey, GDa, GC, GDb, F, Output')
STEP_STRUCT = namedtuple('STEP_STRUCT',
                         'id, num_of_bytes, name')
STEPS_PROVIDED.RoundKey    = STEP_STRUCT(id=0, num_of_bytes=8,
                                         name='RoundKey')
STEPS_PROVIDED.Right       = STEP_STRUCT(id=1, num_of_bytes=8,
                                         name='Right')
STEPS_PROVIDED.AddRoundKey = STEP_STRUCT(id=2, num_of_bytes=8,
                                         name='AddRoundKey')
STEPS_PROVIDED.GDa         = STEP_STRUCT(id=3, num_of_bytes=4,
                                         name='GDa')
STEPS_PROVIDED.GC          = STEP_STRUCT(id=4, num_of_bytes=4,
                                         name='GC')
STEPS_PROVIDED.GDb         = STEP_STRUCT(id=5, num_of_bytes=4,
                                         name='GDb')
STEPS_PROVIDED.F           = STEP_STRUCT(id=6, num_of_bytes=8,
                                         name='F')
STEPS_PROVIDED.Output      = STEP_STRUCT(id=7, num_of_bytes=16,
                                         name='Output')


def get_step_struct(step):
    """
    This method takes input the step (int or str) and returns the namedtuple
    STEP_STRUCT(id, num_of_bytes, name)

    Parameters
    ----------
    step: int str

    Returns
    -------
    STEP_STRUCT

    """

    _step_index = -999

    if type(step) is str:
        if step not in STEPS_PROVIDED._fields:
            raise ValueError(
                SEED_ALGO.header  +
                'Provided step: ' + step + ' not in ' + STEPS_PROVIDED._fields +
                SEED_ALGO.footer
            )
        else:
            _step_index = STEPS_PROVIDED._fields.index(step)
    elif type(step) is int:

        if step < 0 or step > 7:
            raise ValueError(
                SEED_ALGO.header  +
                'Provided step: ' + str(step) + ' not in between 0 to 7' +
                SEED_ALGO.footer
            )
        else:
            _step_index = step

    else:
        raise ValueError(
            SEED_ALGO.header  +
            'Expecting a string or int that describes step in SEED algorithm.' +
            SEED_ALGO.footer
        )

    if _step_index is STEPS_PROVIDED.RoundKey.id:
        return STEPS_PROVIDED.RoundKey
    elif _step_index is STEPS_PROVIDED.Right.id:
        return STEPS_PROVIDED.Right
    elif _step_index is STEPS_PROVIDED.AddRoundKey.id:
        return STEPS_PROVIDED.AddRoundKey
    elif _step_index is STEPS_PROVIDED.GDa.id:
        return STEPS_PROVIDED.GDa
    elif _step_index is STEPS_PROVIDED.GC.id:
        return STEPS_PROVIDED.GC
    elif _step_index is STEPS_PROVIDED.GDb.id:
        return STEPS_PROVIDED.GDb
    elif _step_index is STEPS_PROVIDED.F.id:
        return STEPS_PROVIDED.F
    elif _step_index is STEPS_PROVIDED.Output.id:
        return STEPS_PROVIDED.Output


# documentation
STEPS_PROVIDED.__doc__ += ': \nStructure to store all SEED steps'
STEP_STRUCT.__doc__ += ': \nStructure to store fields for each SEED step'
STEP_STRUCT.id.__doc__ = \
    'Unique id of SEED intermediate step'
STEP_STRUCT.num_of_bytes.__doc__ = \
    'Number of bytes used by SEED intermediate step'
# .......................................................................... end


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
        step : str
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

        # some important checks
        # keysize check
        if keysize not in SEED_ALGO.bits_supported:
            raise ValueError(
                SEED_ALGO.header  +
                'Keysize can be only 128 bit' +
                SEED_ALGO.footer
            )
        # step check
        if step not in STEPS_PROVIDED._fields:
            raise ValueError(
                SEED_ALGO.header  +
                'Provided step: ' + step + ' not in ' + STEPS_PROVIDED._fields +
                SEED_ALGO.footer
            )


        self.rnd = rnd
        self.step = step
        self.keysize = keysize
        self.decrypt = decrypt

        # Determine num_val, the number of values a single element of the
        # intermediate might take. This is constant (2**8) in case of SEED as
        # every element of intermediate step is made up of 8 bits.
        num_val = SEED_ALGO.num_val

        # max_range, or the number of elements in each intermediate-array.
        # This is not constant and can take different value based on step
        # selected.
        max_range = STEPS_PROVIDED._asdict()[step]
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
        # todo: later
        # super().__init__(trace_file, max_range, num_val, local)





















