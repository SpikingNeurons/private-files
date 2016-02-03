# -*- coding: utf-8 -*-
"""
Implementation of DES and its CryptoDataProcessor, CryptoDatum and
CryptoTarget.
DES intermediates are as follows:
16 rounds of:
    RoundKey (the key used for the round)
    Right (the right column of the Feistel scheme)
    AddRoundKey (the XOR of the round key with the state)
    Feistel (the Feistel function)
One final intermediate:
    Output

String identifying intermediates are constructed like so:
'<round>:<step>'
Thus, a valid intermediate might be: '5:Feistel', or '12:AddRoundKey'.
'16:Output' is the only intermediate containing the word 'Output',
as there can't be an output of the algorithm in a round before the last one.

TODO
use http://www.billstclair.com/grabbe/des.htm
"""

import numpy as np
from . import CryptoDataProcessor, CryptoDatum, CryptoTarget

import logging
logger = logging.getLogger(__name__)

# number of elements in a single intermediate
# (DES works largely focused on individual bits
#  but we still store information in arrays of integers,
#  and we will always store 8 for a single intermediate)
DES_MAX_RANGE = 8

# NOTE: all these lookup tables are different from the ones found
# on, say, wikipedia. you'll notice each value here is decreased by one.
# this is because indeces generally start at 0!

# the initial permutation table
IP = [57, 49, 41, 33, 25, 17,  9,  1,
      59, 51, 43, 35, 27, 19, 11,  3,
      61, 53, 45, 37, 29, 21, 13,  5,
      63, 55, 47, 39, 31, 23, 15,  7,
      56, 48, 40, 32, 24, 16,  8,  0,
      58, 50, 42, 34, 26, 18, 10,  2,
      60, 52, 44, 36, 28, 20, 12,  4,
      62, 54, 46, 38, 30, 22, 14,  6]

# the final permutation table;
# it's the inverse of the initial permutation table!
FP = [39,  7, 47, 15, 55, 23, 63, 31,
      38,  6, 46, 14, 54, 22, 62, 30,
      37,  5, 45, 13, 53, 21, 61, 29,
      36,  4, 44, 12, 52, 20, 60, 28,
      35,  3, 43, 11, 51, 19, 59, 27,
      34,  2, 42, 10, 50, 18, 58, 26,
      33,  1, 41,  9, 49, 17, 57, 25,
      32,  0, 40,  8, 48, 16, 56, 24]

# the 'expansion function', really just a lookup table,
# but with some duplicate entries, thus expanding 32 bits into 48
E = [31,  0,  1,  2,  3,  4,
     3,   4,  5,  6,  7,  8,
     7,   8,  9, 10, 11, 12,
     11, 12, 13, 14, 15, 16,
     15, 16, 17, 18, 19, 20,
     19, 20, 21, 22, 23, 24,
     23, 24, 25, 26, 27, 28,
     27, 28, 29, 30, 31,  0]

# Permutation table;
# used to shuffle the bits of a 32-bit half-block
P = [15,  6, 19, 20, 28, 11, 27, 16,
      0, 14, 22, 25,  4, 17, 30,  9,
      1,  7, 23, 13, 31, 26,  2,  8,
     18, 12, 29,  5, 21, 10,  3, 24]

# Permuted choice 1
# The "Left" and "Right" halves of the table show which bits from
# the input key form the left and right sections of the key schedule state.
# Note that only 56 bits of the 64 bits of the input are selected;
# the remaining eight (8, 16, 24, 32, 40, 48, 56, 64) were specified
# for use as parity bits.
PC1_left = [56, 48, 40, 32, 24, 16,  8,
             0, 57, 49, 41, 33, 25, 17,
             9,  1, 58, 50, 42, 34, 26,
            18, 10,  2, 59, 51, 43, 35]

PC1_right = [62, 54, 46, 38, 30, 22, 14,
              6, 61, 53, 45, 37, 29, 21,
             13,  5, 60, 52, 44, 36, 28,
             20, 12,  4, 27, 19, 11,  3]

# Permuted choice 2
# This permutation selects the 48-bit subkey for each round
# from the 56-bit key-schedule state.
PC2 = [13, 16, 10, 23,  0,  4,  2, 27,
       14,  5, 20,  9, 22, 18, 11,  3,
       25,  7, 15,  6, 26, 19, 12,  1,
       40, 51, 30, 36, 46, 54, 29, 39,
       50, 44, 32, 47, 43, 48, 38, 55,
       33, 52, 45, 41, 49, 35, 28, 31]

# S-Boxes
# DES uses 8 different substitution boxes
S1 = [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
       0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
       4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
      15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13]

S2 = [15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
       3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
       0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
      13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9]

S3 = [10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
      13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
      13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
       1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12]

S4 = [ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
      13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
      10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
       3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14]

S5 = [ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
      14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
       4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
      11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3]

S6 = [12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
      10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
       9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
       4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13]

S7 = [ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
      13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
       1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
       6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12]

S8 = [13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
       1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
       7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
       2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11]

def to_int(l):
    """ Turn a list of booleans into an integer.
    """

    # create a string representing the boolean number from the list,
    # then cast it to int
    # (note the '2' as the second parameter to int(), specifying base 2!)
    return int('0b' + ''.join(['1' if e else '0' for e in l]), 2)

def to_bin_list(i, width):
    """ Turn an integer into a list of booleans,
        which represent its value in binary.
    """

    # turn i into binary (a string), and strip away the first two chars,
    # which are always '0b', the identifier that it's in fact binary
    s = bin(i)[2:]

    # make sure we always create at least <width> bits
    while len(s) < width:
        s = '0' + s

    return [True if int(e) else False for e in s]

def to_bin_list_from_list(l, width):
    """ Turn a list of integers into a list of booleans with the desired
        length (if possible)
    """

    ret = []
    deficit = 0
    length_per_int = width / len(l)

    for elem in l:
        b_elem = to_bin_list(elem, 0)
        deficit += length_per_int - len(b_elem)

        # while we're at a deficit, keep inserting 0s (or rather Falses)
        # until we match the length we need
        while deficit < 0:
            b_elem.insert(0, False)

        ret += b_elem

    return ret

def apply_sbox(sbox, l):
    """ Apply the given substitution-box to the list.
        List 'l' must have a length of 6 and only consist of 1s and 0s,
        i.e. booleans.
    """

    assert len(l) == 6, "Invalid length: {}".format(len(l))

    if l[0] == 0 and l[5] == 0:
        row = 0
    elif l[0] == 0 and l[5] == 1:
        row = 1
    elif l[0] == 1 and l[5] == 0:
        row = 2
    else:
        row = 3

    col = to_int(l[1:5])

    ret = sbox[(row * 16) + col]
    ret = to_bin_list(ret, 4)

    return ret

# Before the round subkey is selected, each half of the key schedule state
# is rotated left by a number of places.
# This table specifies the number of places rotated.
rotation_schedule = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

def rotate(l, steps):
    """ Rotate a list to the left by a number of steps.
    """

    assert steps < len(l), "This function isn't smart enough to "\
                           "rotate a list by more steps than it is long "\
                           "(because it doesn't have to be right now)."

    if len(l) <= 1:
        return l

    end = l[:steps]
    l = l[steps:]
    return l + end

def get_subkey(key, rnd):
    """ Compute the subkey for the specified round from the key.
    """

    assert 1 <= rnd <= 16, "Invalid round!"
    assert len(key) == 8

    key_bits = []
    for elem in key:
        key_bits += to_bin_list(elem, 0)

    print(len(key_bits))
    if len(key_bits) == 56:
        # insert some parity bits so our choice boxes still work
        for i in range(7, 71, 8):
            key_bits.insert(i, 0)

    assert len(key_bits) == 64

    key_left = []
    key_right = []

    for elem in PC1_left:
        key_left.append(key_bits[elem])

    for elem in PC1_right:
        key_right.append(key_bits[elem])

    assert len(key_left) == 28
    assert len(key_right) == 28

    for i in range(rnd):
        key_left = rotate(key_left, rotation_schedule[rnd])
        key_right = rotate(key_right, rotation_schedule[rnd])

    temp = key_left + key_right

    ret = []

    for elem in PC2:
        ret.append(temp[elem])

    return ret

def AddRoundKey(prev_right, subkey):
    """ Perform the AddRoundKey operation;
        usually this is part of the F or Feistel function,
        but we want it to be a separate intermediate.
        Besides the XOR of the previous right intermediate,
        it also consists of the expansion of the input.
    """

    assert prev_right.shape[1] == 8
    # assert max(prev_right) < 2 ** 4, "Elements must be less than 2 ** 4!"

    out = np.copy(prev_right)

    # turn the list of bytes into a list of bits
    subkey_bits = []
    for elem in subkey:
        subkey_bits += to_bin_list(elem, 6)

    for i in range(len(out)):
        # turn the list of bytes into a list of bits
        right_bits = []
        for elem in prev_right[i]:
            right_bits += to_bin_list(elem, 4)

        # expand the 32 bits into 48 bits using the E table
        exp_right_bits = []
        for elem in E:
            exp_right_bits.append(right_bits[elem])

        # add (xor) the subkey to the expanded block
        ark = []
        for e, k in zip(exp_right_bits, subkey_bits):
            ark.append(e ^ k)

        assert len(ark) == 48
        # transform the list of bits back into a list of integers
        # of size 8
        ret = []
        for i in range(0, 48, 6):
            ret.append(to_int(ark[i:i+6]))

        assert len(ret) == 8

        out[i] = ret

    return out

def Feistel(ark):
    """ Perform the Feistel-step; in our implementation, this differs slightly
        from the original specification. Because we want to have a value
        as an intermediate that is located within Feistel (AddRoundKey),
        we out-sourced some of Feistel's functionality to the AddRoundKey
        function.
    """

    assert ark.shape[1] == 8
    # assert max(ark) < 2 ** 6, "Items in ark must hold no more than 6 bits "\
    #                           "worth of information!"

    out = np.copy(ark)

    for i in range(len(out)):
        # create a list of bits from our list of 6-bit bytes
        ark_bits = []
        for elem in ark[i]:
            ark_bits += to_bin_list(elem, 6)

        # should end up with 48 bits
        assert len(ark_bits) == 48

        ### Substitution ###
        # apply the different sboxes to the different sections
        # make a temp list with 32 entries to save results in
        temp = [0 for i in range(32)]
        temp[0:4] = apply_sbox(S1, ark_bits[0:6])
        temp[4:8] = apply_sbox(S2, ark_bits[6:12])
        temp[8:12] = apply_sbox(S3, ark_bits[12:18])
        temp[12:16] = apply_sbox(S4, ark_bits[18:24])
        temp[16:20] = apply_sbox(S5, ark_bits[24:30])
        temp[20:24] = apply_sbox(S6, ark_bits[30:36])
        temp[24:28] = apply_sbox(S7, ark_bits[36:42])
        temp[28:32] = apply_sbox(S8, ark_bits[42:48])

        ### Permutation ###
        f = []
        for elem in P:
            f.append(ark_bits[elem])

        assert len(f) == 32

        ret = []
        for i in range(0, 32, 4):
            ret.append(to_int(f[i:i+4]))

        assert len(ret) == 8
        assert max(ret) < 2 ** 4

        out[i] = ret

    ### Finished! ###
    return out

def Right(prev_right, feistel):
    """ The 'Right' step computes the block that will be used as
        the next right columns of the Feistel scheme. It is the XOR
        of the F-function output and the previous right column block.
    """

    ret = np.bitwise_xor(prev_right, feistel)

    assert len(ret) == 8
    assert max(ret) < 2 ** 4

    return ret

def Output(left, right):
    """ Calculate the final output of DES using the left and right
        columns of the Feistel scheme.
    """

    left_bits = to_bin_list(left, 8)
    right_bits = to_bin_list(right, 8)

    out_bits = left_bits + right_bits

    assert len(out_bits) == 64

    out = []
    for elem in FP:
        out.append(out_bits[elem])

    assert len(out) == 64

    ret = []
    for i in range(0, 64, 8):
        ret.append(to_int(out[i:i+8]))

    assert len(ret) == 8

    return ret

def DES_round(left_in, right_in, key, rnd):
    """ Perform one round of DES on the two inputs, each 32 bits in size.
        Other inputs are the key and the round number.
    """

    left_out = right_in
    right_out = [l ^ r for l, r in zip(left_in, Feistel(right_in, key, rnd))]

    return left_out, right_out

class DESCryptoTarget(CryptoTarget):
    """ Holds only information about the one intermediate value,
        but no computed information.
        Used as keys for the CryptoDataProcessor's dictionary.
    """

    def __init__(self, rnd, step, decrypt):
        self.rnd = rnd
        self.step = step
        self.decrypt = decrypt

        # max_range, or the number of elements in each intermediate-array,
        # is constant, just like in AES. Only here we have 8 array entries,
        # instead of AES' 16
        self.max_range = DES_MAX_RANGE

        # determine num_val, the number of values
        # a single element of this intermediate might take
        if step == 'RoundKey':
            # a RoundKey is 48 bits in size; thus, if it's divided into
            # 8 pieces, each piece can represent 2 ** 6 different values
            self.num_val = 2 ** 6
        elif step == 'Right':
            # Right has 32 bits, so each piece of data can hold 2 ** 4 values
            self.num_val = 2 ** 4
        elif step == 'AddRoundKey':
            # 48 bits again
            self.num_val = 2 ** 6
        elif step == 'Feistel':
            # 32 bits
            self.num_val = 2 ** 4
        elif step == 'Output':
            # and Output holds all of the 64 bits, so each piece
            # in the array can fill up a whole byte, or 8 bits,
            # thus being able to take one of 256 different values.
            self.num_val = 2 ** 8
        else:
            raise ValueError("Invalid step!")

    def __str__(self):
        return str(self.rnd) + ':' + self.step

    def __hash__(self):
        return hash((self.rnd, self.step, self.decrypt))

    def __eq__(self, other):
        """ Overwrite equality-check in order to be able to properly
            use DESCryptoTarget as a dictionary key.
        """

        # NOTE: make sure to always do the type check first!
        return type(self) == type(other)\
           and (self.rnd == other.rnd)\
           and (self.step == other.step)\
           and (self.decrypt == other.decrypt)

    @property
    def str(self):
        return self.__str__()

class DESCryptoDatum(CryptoDatum):
    """ Holds exactly one intermediate value and some meta-data
        for the DES algorithm.
        The CryptoDataProcessor uses these to store data.
    """

    def __init__(self, rnd, step, inpt, decrypt, key=None):

        self._round = rnd
        self._step = step
        self._decrypt = decrypt

        if key is None and step is 'AddRoundKey':
            raise ValueError("'AddRoundKey' step requires a key!")

        if self._step == 'AddRoundKey':
            self._datum = AddRoundKey(inpt, key)
        elif self._step == 'Right':
            assert type(inpt) is tuple
            self._datum = Right(inpt[0], inpt[1])
        elif self._step == 'Feistel':
            self._datum = Feistel(inpt)
        elif self._step == 'Output':
            assert type(inpt) is tuple
            self._datum = Output(inpt[0], inpt[1])
        else:
            raise ValueError("Invalid step!")

    def __str__(self):
        return str(self._round) + ':' + self._step

class DESCryptoDataProcessor(CryptoDataProcessor):
    """ The DES-specific CryptoDataProcessor.
        Provides the different intermediate values that
        result from DES-encryption or -decryption.
    """

    def __init__(self, trace_file, decrypt=False):

        # if the key in the trace file is a local field, it means that
        # each trace has its own unique key; thus we set multikey to True:
        if trace_file.has_local_field('key'):
            self._multikey = True
        else:
            self._multikey = False

        # keysize for DES is always the same.
        self._keysize = 64

        if type(decrypt) != bool:
            raise TypeError("Invalid parameter: decrypt must be bool!")

        self._decrypt = decrypt

        # initialize data as empty dictionary
        self._data = {}

        # DES always consists of 16 rounds
        self.rounds = 16

        # DES consists of these steps:
        self.steps = ['AddRoundKey', 'Feistel', 'Right', 'Output']
        # save our trace file; we'll need it!
        self.trace_file = trace_file

        # initialize trace range to None; will be updated upon first request
        # to __getitem__.
        # We need to keep track of this because we'll only ever compute data
        # within this trace range, thus saving computation time and memory!
        self._trange = None

    def __check_target(self, target):
        """ Check wether the given DESCryptoTarget object is valid for this
            processor.
        """

        if target.rnd <= 0 or target.rnd > self.rounds:
            return False

        if target.step not in self.steps:
            return False

        if target.step == 'Output' and target.rnd != self.rounds:
            return False

        if target.decrypt != self._decrypt:
            return False

        return True

    def __normalize_datum(self, datum):
        """ Normalize the given datum into a tuple of round and step.
            Along the way, check it validity.
        """

        if isinstance(datum, DESCryptoTarget):
            return datum.rnd, datum.step

        # split the datum, which should be a string, along the colon:
        try:
            args = datum.split(':')
        # if we catch an AttributeError, re-raise it as a more appropriate
        # TypeError.
        except AttributeError:
            raise TypeError("Invalid type! Expected str but got {}!"
                            .format(type(datum)))

        # the tuple 'args' should now hold exactly two elements:
        if len(args) != 2:
            raise ValueError("Invalid datum! Expected a shape like: "
                             "'<round>:<step>'!")

        try:
            rnd = int(args[0])
            step = args[1]
        except ValueError:
            raise ValueError("Invalid datum! Expected a shape like: "
                             "'<round>:<step>'!")

        if rnd not in range(1, self.rounds + 1):
            if rnd < 0:
                rnd = self.rounds + (rnd + 1)
                if rnd <= 0:
                    raise IndexError("Invalid round! Expected value "
                                     "between 1 and 16!")
            else:
                raise IndexError("Invalid round! Expected value "
                                 "between 1 and 16!")

        if step not in self.steps:
            raise IndexError("Invalid step! Expected one of: {}"
                             .format(self.steps))

        if step == 'Output' and rnd != self.rounds:
            raise IndexError("'Output' step must only occur on last round!")

        return (rnd, step)

    def provides(self, query):
        """ Check wether this CPD can provide the requested field.
        """

        if isinstance(query, DESCryptoTarget):
            return self.__check_target(query)

        try:
            _, _ = self.__normalize_datum(query)
        except (TypeError, ValueError, IndexError):
            return False

        return True

    def _fetch_input(self, target):
        """ Fetch the required input for the specified datum.
            This is, of course, the output of the previous datum,
            or the plaintext/ciphertext if it's for the first round.
        """

        rnd, step = target.rnd, target.step

        if step == 'Output' and rnd != self.rounds:
            raise ValueError("Bad request: 'Output' step only "
                             "occurs in last round!")

        if rnd == 1 and step == 'AddRoundKey':
            # for the very first step, we start with the plaintext
            return self.trace_file['plaintext', self._trange]
        elif step == 'Right':
            prev_right = DESCryptoTarget(rnd - 1, 'Right', self._decrypt)
            return (self[prev_right, None, None],
                    self[self.get_previous_target(target), None, None])
        elif step == 'Output':
            # create two static targets; final Output always requires
            # round 15's right column and round 16's right column
            right15_target = DESCryptoTarget(15, 'Right', self._decrypt)
            right16_target = DESCryptoTarget(16, 'Right', self._decrypt)

            # and return them as a tuple
            return (self[right15_target, None, None],
                    self[right16_target, None, None])

        return self[self.get_previous_target(target), None, None]

    def _compute(self, target, key=None):
        """ Compute the requested datum by creating a CryptoDatum for it.
        """

        rnd, step = target.rnd, target.step

        inpt = self._fetch_input(target)

        if key is None and step == 'AddRoundKey':
            if not self._decrypt:
                key = get_subkey(self.trace_file['key'], rnd)
            else:
                raise NotImplementedError()

        self._data[target] = DESCryptoDatum(rnd, step, inpt,
                                            decrypt=self._decrypt,
                                            key=key)

    def __getitem__(self, key):
        """ Return the requested datum;
            If necessary, compute it first.
            Raise an exception if the requested datum is invalid.
            'key' must be a tuple with exactly three elements:
            a DESCryptoTarget, a trace-range and a data-range.
        """

        if not isinstance(key, tuple):
            raise TypeError("'key'-parameter must be a tuple!")

        if len(key) != 3:
            raise ValueError("'key'-parameter must be a tuple of length 3!")

        if not isinstance(key[0], DESCryptoTarget):
            raise ValueError("First element of 'key' must be of type "
                             "DESCryptoTarget!")

        target = key[0]
        trace_range = key[1]
        data_range = key[2]

        if target.decrypt != self._decrypt:
            raise ValueError("Target not compatible! 'Decrypt'-mismatch!")

        # if a different trace range from the one we saved is given,
        # clear our data!
        if trace_range is not None and self._trange != trace_range:
            self._data = {}

        # process different possibilities for trace_range
        if trace_range is slice(None):
            # assume all traces are wanted
            self._trange = slice(0, len(self.trace_file))
        elif type(trace_range) is int:
            # assume only that one trace is wanted, but make a slice out of
            # it to maintain compatibility
            self._trange = slice(trace_range, trace_range + 1)
        elif type(trace_range) is slice:
            # if it's only a slice, we can keep it the way it is
            self._trange = trace_range
        elif trace_range is None:
            # trace_range will only be None if called from our own _compute;
            # the TraceFileHandler will only ever use slice(None) is the slice
            # is empty. If we get a None for trace_range, it's because we do
            # not want it to be applied multiple times, and this None will be
            # a signal that it has already been applied before.
            # So, do nothing:
            pass
        else:
            raise TypeError("Invalid trace range!")

        # attempt to return the requested datum straight away:
        try:
            if data_range is None:
                return self._data[target].datum
            else:
                return self._data[target].datum[:,data_range]
        # catch the KeyError that occurs if it's not found...
        except KeyError:
            # ...and try to compute it
            try:
                self._compute(target)
            # if this fails, too, it might mean we received a request for a
            # round key, so we try that:
            except ValueError:
                rnd, step = target.rnd, target.step
                if step == 'RoundKey':
                    return get_subkey(self.trace_file['key'], rnd)
                else:
                    raise KeyError("Item not provided by this processor!")
            # if we except anything besides a ValueError, re-raise that
            # as a more appropriate KeyError:
            except (TypeError, IndexError):
                raise KeyError("Item not provided by this processor!")

        # if we made it this far, the computation finished without raising
        # exceptions, so we can safely assume that our datum is now saved in
        # self._data, and we can return it
        ret = self._data[target].datum if data_range is None \
              else self._data[target].datum[:,data_range]

        return ret

    def get_crypto_target(self, datum):
        """ Return a DESCryptoTarget object holding information about
            the datum, which does not include the computed array, but
            some of its properties and meta-data.
        """

        if isinstance(datum, DESCryptoTarget):
            if self.__check_target(datum):
                return datum
            else:
                raise ValueError("Invalid target for this processor!")

        rnd, step = self.__normalize_datum(datum)
        ret = DESCryptoTarget(rnd, step, self._decrypt)

        if self.__check_target(ret):
            return ret
        else:
            raise ValueError("Invalid target for this processor!")

    def get_previous_target(self, target):
        if target.decrypt != self._decrypt:
            raise ValueError("Incompatible target! Decrypt mismatch!")

        rnd, step = None, None

        if target.step == 'AddRoundKey':
            if target.rnd == 1:
                raise ValueError("No previous target!")
            else:
                rnd = target.rnd - 1
                step = 'Right'
        elif target.step == 'Feistel':
            rnd = target.rnd
            step = 'AddRoundKey'
        elif target.step == 'Right':
            rnd = target.rnd
            step = 'Feistel'
        elif target.step == 'Output':
            rnd = target.rnd
            step = 'Right'

        if rnd is None or step is None:
            raise ValueError("Invalid target!")

        return DESCryptoTarget(rnd, step, self._decrypt)

    def get_previous_datum(self, datum):
        rnd, step = self.__normalize_datum(datum)
        target = DESCryptoTarget(rnd, step, self._decrypt)
        return str(self.get_previous_target(target))

    def get_subkey(self, rnd):
        """ Return the subkey for the specified round.
        """

        # enable wrap-around of index
        if rnd < 0:
            rnd = self.rounds + (rnd + 1)

        assert rnd > 0 and rnd <= self.rounds, "Invalid round!"

        # get_subkey is also a function outside this class;
        # TODO: change names around to make this less confusing!
        # NOTE: will probably be fixed once the cython implementation
        # is done...
        return get_subkey(self.trace_file['key'], rnd)

    @property
    def crypto_targets(self):
        """ Return a list of all valid fields this processor provides.
        """

        ret = []
        for i in range(1, self.rounds):
            for elem in ['AddRoundKey', 'Feistel', 'Right']:
                ret.append(str(i) + ':' + elem)
        ret.append('16:Output')

        return ret

    @staticmethod
    def get_datum_string(rnd, step):
        return str(rnd) + ':' + step

    def guess_data(self, target, guess, trace_range, data_range=None):
        pass
