# -*- coding: utf-8 -*-
"""
This module implements the Advanced Encryption Standard,
as well as a CryptoDataProcessor and CryptoDatum to make use
of its functionality.
AES128 is implemented as such:
9 rounds of:
    AddRoundKey,
    SubBytes,
    ShiftRows,
    MixColumns
1 round of:
    AddRoundKey,
    SubBytes
    ShiftRows
1 final step:
    AddRoundKey

(Warning: AES192 and AES256 currently not supported!)

AES192 and AES256 will be implemented analogously,
with a total of 12 and 14 rounds, respectively, and also always
with an extra step of AddRoundKey at the end.
The CryptoDataProcessor is used by a TraceFileHandler;
if a datum is requested from the TraceFileHandler, and its processor can
provide it, it is fetched from it and returned.
AES data can be requested as follows:
Say we have a TraceFile object:
  tf = TraceFile('path/to/tracefile.bin')

We create a TraceFileHandler:
  tfh = TraceFileHandler(tf, algo='AES')

Now, we can request cryptographic intermediates:
  some_variable = tfh['4:SubBytes']

Intermediate-requests, as you can see, are done using the __getitem__ function.
The key in the request (the '[]') can be a string of the following format:
<round>:<step>
as seen in the above example, or an object of AESCryptoTarget, like so:
  target = AESCryptoTarget(4, 'SubBytes', 128, False)
  some_variable = tfh[target]

The 128 in the instantiation refers to the keysize, and the False means
decryption is turned off. Usually, a user will use a string for convenience;
internally the CryptoDataProcessor only accepts CryptoTargets, while the
TraceFileHandler accepts both.
The round can be any number between 1 and 10 for AES128, and, analogously,
1 and 12 or 1 and 14 for AES192 and AES256, respectively.
We also allow negative input for the round; this works just like Python's
list-indexing: index [-1] is the last item of the list. Similarly,
round [-1] would be the last round of AES.
Thus, ['-1:FinalAddRoundKey'] is a valid request!
The step must be one of the steps listed above, to fetch an intermediate,
or 'FinalAddRoundKey' to fetch the fully encrypted ciphertext.
This corresponds to the final AddRoundKey step, which would techincally
belong to round 10, but then we'd have two possible return values for a
request to '10:AddRoundKey'.

Decryption:
We now support decryption; this means an AESCryptoDataProcessor can be
instantiated with decrypt=True and will then only allow fetching of
decryption-intermediates. The order of intermediates in AES-128 looks like this:
1 round of:
    AddRoundKey
    ShiftRows
    SubBytes
9 rounds of:
    AddRoundKey
    MixColumns
    ShiftRows
    SubBytes
1 final step:
    AddRoundKey

The final AddRoundKey-step will yield the plaintext.
To avoid confusion with the other AddRoundKey-steps, this final step
is also called "10:FinalAddRoundKey".
The algorithms to compute each intermediate are not, in fact, the exact same
ones as used in the encryption process, but rather their inverses.

Please feel free to contact tronje.krabbe@nxp.com with any questions or
bug-reports, or even feature-requests.
"""

import numpy as np

from . import CryptoDataProcessor, CryptoDatum, CryptoTarget

from . import AES_helper
from .T_tables import T_tables


# import and instantiate logging functionality
import logging
logger = logging.getLogger(__name__)

# number elements in a single intermediate
# (in AES equal to number of bytes, of course)
AES_MAX_RANGE = 16

# number of values a value in an intermediate can take
# (in AES, each value being one byte, it's of course
#  2 to the power of 8)
AES_NUM_VAL = 256

# substitution box used in SubBytes step
S_Box = [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
         0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
         0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
         0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
         0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
         0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
         0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
         0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
         0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
         0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
         0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
         0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
         0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
         0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
         0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
         0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
         0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
         0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
         0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
         0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
         0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
         0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
         0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
         0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
         0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
         0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
         0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
         0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
         0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
         0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
         0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
         0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]

# S_Box inverse used in inverse of SubBytes step, needed for decryption
Inv_S_Box = [0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
             0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
             0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
             0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, 
             0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
             0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e, 
             0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
             0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25, 
             0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
             0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, 
             0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
             0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, 
             0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
             0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, 
             0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
             0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 
             0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
             0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, 
             0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
             0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, 
             0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
             0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 
             0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
             0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, 
             0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
             0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, 
             0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
             0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 
             0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
             0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61, 
             0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
             0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]

# lookup-table for determining round constant (Rcon for short)
Rcon = [0x8d, 0x01, 0x02, 0x04,
        0x08, 0x10, 0x20, 0x40,
        0x80, 0x1b, 0x36, 0x6c,
        0xd8, 0xab, 0x4d, 0x9a]

def print_hex(arr):
    """ handy function to print an array of 8-bit integers
        in hexadecimal form, in a neatly arranged fashion.
    """

    for elem in arr:
        s = hex(elem)[2:]
        if len(s) == 1:
            s = '0' + s
        print(s, end=' ')
    print('\n')

def key_schedule_core(inpt, i):
    """ The operation of the Rijndael key schedule.
        It receives a 32-bit word (inpt) and an iteration number (i).
        inpt shall be a list or numpy array of four 8bit hex numbers!
    """

    assert len(inpt) == 4, "inpt does not have correct size!"
    for elem in inpt:   
        assert type(elem) is np.uint8, "element in inpt does not " \
                                       "have correct type (numpy.uint8)!"
        assert elem <= 0xff, "element in inpt is too large!"

    out = []

    # rotate input one byte to the left and save as 'out'
    out.append(inpt[1])
    out.append(inpt[2])
    out.append(inpt[3])
    out.append(inpt[0])

    # apply Rijndael's SBox
    for j in range(len(out)):
        out[j] = S_Box[out[j]]

    # for just the first byte, add an rcon value (XOR)
    out[0] = out[0] ^ Rcon[i]

    assert len(out) == 4, "something went horribly wrong: out has invalid length!"
    return out

def key_schedule(key):
    """ Rijndael's key schedule. Performs key-expansion on the given key;
        returns a numpy.ndarray containing the key at index 0,
        and the required amount of round-keys at the corresponding
        higher indices.
        Note: for performance-reasons, the Cython implementation is used,
        this just lives here as a proof-of-concept.
    """

    # assertions are cool, because we don't catch assertion errors anywhere,
    # which means we know exactly where something goes wrong!
    assert type(key) is np.ndarray, "Invalid key type!"
    assert key.dtype.type is np.uint8, "Invalid data type of key array!"

    # compute keysize in bits
    keysize = len(key) * 8

    # keysize must be one of these; there are no AES versions for other keysizes!
    assert keysize in [128, 192, 256], "Invalid keysize!"

    # declare some constants as specified in en.wikipedia.org/wiki/Rijndael_key_schedule
    ## n is one of [16,24,32] depending on the keysize
    n = {128:16, 192:24, 256:32}[keysize]

    ## b is one of [176,208,240] depending on the keysize
    b = {128:176, 192:208, 256:240}[keysize]

    # initialize current keysize (in bytes!!)
    # use // to do integer division; we don't want this to be a float
    curr_size = keysize // 8

    # initialize our output array;
    # holds 11, 13 or 15 roundkeys of size n each (depending on keysize)
    # NOTE: currently we just use a list, which we later use to create a
    # numpy array, which is then reshaped. Makes some computation easier
    # out = np.zeros((b//16,n), dtype=np.int32)
    # out = []
    out = np.array([], dtype=np.uint8)

    # first n bytes of expanded key are equal to the encryption key
    # out[0] = key
    for elem in key:
        # out.append(elem)
        out = np.append(out, elem)

    # rcon iteration value is set to 1
    rcon_i = 1

    while(curr_size < b):
        # assign the last four bytes of our expanded key to a temporary variable t
        # t = out[rcon_i - 1][-4:]
        t = out[-4:].astype(np.uint8)

        # every n sets, do a complex operation
        if curr_size % n == 0:
            # perform the core operation on t
            t = key_schedule_core(t, rcon_i)
            # increment rcon iteration value
            rcon_i += 1

        # for 256-bit keys, we need to do an extra s-box application
        if keysize == 256 and (curr_size % n) == 16:
            for a in range(4):
                t[a] = S_Box[t[a]]

        # exclusive-OR t with some elements of old key
        for a in range(4):
            # out.append(out[curr_size - n] ^ t[a])
            out = np.append(out, out[curr_size - n] ^ t[a]).astype(np.uint8)
            curr_size += 1

    # create a numpy array with appropriate dimensions out of our list...
    # out = np.array(out, dtype=np.int32).reshape(b // 16, 16)
    out = out.reshape(b // 16, 16)

    # ... and return it!
    return out


def key_schedule_multi(key_sched, rnd, keysize):
    """ key scheduler designed to output a key schedule incrementally.
        Note: for performance-reasons, the Cython implementation is used,
        this just lives here as a proof-of-concept.
    """

    # assertions are cool, because we don't catch assertion errors anywhere,
    # which means we know exactly where something goes wrong!
    assert type(key_sched) is np.ndarray, "Invalid key type!"
    assert key_sched.dtype.type is np.uint8, "Invalid data type of key array!"

    # key_sched should come initialized with the key as the first
    # subkey; therefore, it will be ready for rnd == 1, and therefore
    # this function is intended to be called for rnd >= 2!
    assert rnd > 1, "Invalid round! Round key for 1st round is simply the key!"

    # keysize must be one of these; there are no AES versions for other keysizes!
    assert keysize in [128, 192, 256], "Invalid keysize!"

    # declare some constants as specified in en.wikipedia.org/wiki/Rijndael_key_schedule
    ## n is one of [16,24,32] depending on the keysize
    n = {128:16, 192:24, 256:32}[keysize]

    ## b is one of [176,208,240] depending on the keysize
    b = {128:176, 192:208, 256:240}[keysize]

    # initialize current keysize (in bytes!!)
    # each round key is 16 bytes in size, meaning in the first round,
    # we have a schedule of size 0, in the second of size 16, then 32, ...
    curr_size = (rnd - 1) * 16

    # initialize our output array
    # make it 1-dimensional for easier processing
    out = key_sched.reshape(key_sched.shape[0] * key_sched.shape[1])
    assert out.size == key_sched.size

    # rcon iteration value is set
    rcon_i = curr_size // n

    while curr_size < rnd * 16:
        # assign the last four bytes of our expanded key to a temporary variable t
        t = []
        for a in range(4):
            t.append(out[a + curr_size - 4])

        # every n sets, do a complex operation
        if curr_size % n == 0:
            # perform the core operation on t
            t = key_schedule_core(t, rcon_i)
            # increment rcon iteration value
            rcon_i += 1

        # for 256-bit keys, we need to do an extra s-box application
        if keysize == 256 and (curr_size % n) == 16:
            for a in range(4):
                t[a] = S_Box[t[a]]

        # exclusive-OR t with some elements of old key
        for a in range(4):
            out[curr_size] = out[curr_size - n] ^ t[a]
            curr_size += 1

    # reshape our output to make it 2-dimensional again...
    assert out.size == (b // 16) * 16
    out = out.reshape(b // 16, 16)

    # ... and return it!
    return out


# the following are the bread-and-butter operations of AES;
# for performance reasons, these are not used here, and really only
# exist as backups should the Cython version break at some point.
# Modifications on the Cython code can be checked for correctness
# by using these as references.

def AddRoundKey(inpt, key):
    """ Perform the AddRoundKey step of the AES
        algorithm.
    """

    assert inpt.dtype.type is np.uint8, "Invalid input type!"
    assert key.dtype.type is np.uint8, "Invalid key type!"

    out = np.copy(inpt)

    # check if the key is multi-dimensional
    # if it is, every trace has its own key,
    # which we then have to take into account
    if len(key.shape) == 2:
        assert len(out) == len(key), "Input/key mismatch!"
        for i in range(len(out)):
            for j in range(len(key[0])):
                out[i][j] = out[i][j] ^ key[i][j]
    else:
        for i in range(len(out)):
            for j in range(len(key)):
                out[i][j] = out[i][j] ^ key[j]

    return out

def InvAddRoundKey(inpt, key):
    """ Perform the inverse operation to AddRoundKey,
        which is its own inverse. This function exists solely for clarity!
    """
    return AddRoundKey(inpt, key)

def SubBytes(ARK):
    """ Perform the SubBytes step of the AES
        algorithm.
        Input shall be the output of AddRoundKey!
    """

    assert ARK.dtype.type is np.uint8, "Invalid SR type!"

    out = np.zeros((ARK.shape), dtype=ARK.dtype)
    for i in range(len(out)):
        for j in range(len(out[0])):
            # apply s-box substitution to each value
            out[i][j] = S_Box[ARK[i][j]]
    return out

def InvSubBytes(inpt):
    """ Perform the inverse operation to the SubBytes step,
        using the inverted S-Box table.
    """

    assert inpt.dtype.type is np.uint8, "Invalid dtype!"

    out = np.zeros((inpt.shape), dtype=inpt.dtype)
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = Inv_S_Box[inpt[i][j]]
    return out

def ShiftRows(SB):
    """ Perform the ShiftRows step of the AES
        algorithm.
        input shall be the output of SubBytes!
    """

    assert SB.dtype.type is np.uint8, "Invalid ARK type!"

    out = np.zeros((SB.shape), dtype=SB.dtype)
    for i in range(len(out)):
        for j in range(len(out[0])):
            # shift rows around according to AES specifications!
            # this dictionary is not just random! It is provided
            # like this by AES!
            out[i][j] = SB[i][{
                0:0,
                1:5,
                2:10,
                3:15,
                4:4,
                5:9,
                6:14,
                7:3,
                8:8,
                9:13,
                10:2,
                11:7,
                12:12,
                13:1,
                14:6,
                15:11
            }[j]]
    return out

def InvShiftRows(inpt):
    """ Perform the inverse operation to ShiftRows.
    """

    assert inpt.dtype.type is np.uint8, "Invalid dtype!"

    out = np.zeros((inpt.shape), dtype=inpt.dtype)
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = inpt[i][{
                0:0,
                1:13,
                2:10,
                3:7,
                4:4,
                5:1,
                6:14,
                7:11,
                8:8,
                9:5,
                10:2,
                11:15,
                12:12,
                13:9,
                14:6,
                15:3
            }[j]]
    return out

def MixColumns(SR):
    """ Perform the MixColumns step of the AES
        algorithm.
        Input shall be the output of ShiftRows!
    """

    assert SR.dtype.type is np.uint8, "Invalid SB type!"

    MC = []

    for row in SR:
        temp = [
            T_tables[0][row[0]] ^ T_tables[1][row[1]] ^ T_tables[2][row[2]] ^ T_tables[3][row[3]],
            T_tables[0][row[4]] ^ T_tables[1][row[5]] ^ T_tables[2][row[6]] ^ T_tables[3][row[7]],
            T_tables[0][row[8]] ^ T_tables[1][row[9]] ^ T_tables[2][row[10]] ^ T_tables[3][row[11]],
            T_tables[0][row[12]] ^ T_tables[1][row[13]] ^ T_tables[2][row[14]] ^ T_tables[3][row[15]]
        ]
  
        temp = [
            temp[0]>>24, (temp[0]&0xff0000)>>16, (temp[0]&0xff00)>>8, temp[0]&0xff,\
            temp[1]>>24, (temp[1]&0xff0000)>>16, (temp[1]&0xff00)>>8, temp[1]&0xff,\
            temp[2]>>24, (temp[2]&0xff0000)>>16, (temp[2]&0xff00)>>8, temp[2]&0xff,\
            temp[3]>>24, (temp[3]&0xff0000)>>16, (temp[3]&0xff00)>>8, temp[3]&0xff
        ]

        MC.append(temp)

    ret = np.array(MC, dtype=np.uint8)
    return ret

    # 'Textbook' implementation, returns the same results
    # as the above, but is a little slower...

    # ret = np.zeros(SR.shape, dtype=SR.dtype)

    # for i in range(len(SR)):
    #     temp = SR[i]

    #     for j in range(4):
    #         r = temp[j * 4 : j * 4 + 4]
    #         a = [0, 0, 0, 0]
    #         b = [0, 0, 0, 0]

    #         for c in range(4):
    #             a[c] = r[c]
    #             h = r[c] & 0x80
    #             b[c] = r[c] << 1

    #             if h == 0x80:
    #                 b[c] ^= 0x1b

    #         ret[i][j * 4] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1]
    #         ret[i][j * 4 + 1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2]
    #         ret[i][j * 4 + 2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3]
    #         ret[i][j * 4 + 3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]

    # return ret

def gmul(a, b):
    """ Multiply a by b in Rijndael's galois field,
        required by InvMixColumns.
    """

    p = 0

    for count in range(8):
        if (b & 1) == 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set == 0x80:
            a ^= 0x1B
        b >>= 1

    return p

def InvMixColumns(inpt):
    """ The inverse of the MixColumns operation.
        Non-optimized implementation for lack of good, reliable resources;
        TODO: come back to this and optimize!
    """

    ret = np.zeros(inpt.shape, dtype=inpt.dtype)

    for i in range(len(inpt)):
        temp = inpt[i]

        # in this loop, the 'core' functionality of MixColumns
        # is executed; we do this 4 times because MixColumns is designed
        # to be used on 4-byte blocks, but by design we have 16-byte
        # rows to process.
        for j in range(4):
            r = temp[j * 4: j * 4 + 4]
            a = r.copy()

            # the gmul-calls can be optimized away, but that might take
            # a while to figure out and then debug, so we'll leave it at this
            # for now
            ret[i][j * 4 + 0] =\
                gmul(a[0], 14) ^ gmul(a[3], 9) ^ gmul(a[2], 13) ^ gmul(a[1], 11)
            ret[i][j * 4 + 1] =\
                gmul(a[1], 14) ^ gmul(a[0], 9) ^ gmul(a[3], 13) ^ gmul(a[2], 11)
            ret[i][j * 4 + 2] =\
                gmul(a[2], 14) ^ gmul(a[1], 9) ^ gmul(a[0], 13) ^ gmul(a[3], 11)
            ret[i][j * 4 + 3] =\
                gmul(a[3], 14) ^ gmul(a[2], 9) ^ gmul(a[1], 13) ^ gmul(a[0], 11)

    return ret

class AESCryptoTarget(CryptoTarget):
    """ Holds only information about one intermediate value,
        but no actually computed information.
        Used as keys for the CryptoDataProcessor's dictionary,
        among other things.
    """

    # def __init__(self, rnd, step, prev_datum, keysize, decrypt=False):
    def __init__(self, rnd, step, keysize, decrypt):
        self.rnd = rnd
        self.step = step
        self.max_range = AES_MAX_RANGE
        self.num_val = AES_NUM_VAL
        # self.know_input = prev_datum
        self.keysize = keysize
        self.decrypt = decrypt

    def __str__(self):
        return str(self.rnd) + ':' + self.step

    def __hash__(self):
        return hash((self.rnd, self.step, self.keysize, self.decrypt))

    def __eq__(self, other):
        """ Overwrite compare-equality method in order to be able
            to properly use AESCryptoTarget as a dictionary key.
        """

        # NOTE: make sure to always do the type check first!
        return type(self) == type(other)\
           and (self.rnd == other.rnd)\
           and (self.step == other.step)\
           and (self.keysize == other.keysize)\
           and (self.decrypt == other.decrypt)

    @property
    def str(self):
        return self.__str__()

class AESCryptoDatum(CryptoDatum):
    """ Holds one intermediate value and some meta-data for
        the AES algorithm.
        This is used to store data computed by the CryptoDataProcessor,
        with multiple of these in a dictionary, with CryptoTargets as keys.
    """

    # Please note:
    # AES consists of several rounds; variables used in this class
    # that represent a round are usually named 'rnd', because
    # 'round' is a built-in function of Python, and we want to prevent
    # errors related to this!
    # We may also want/need to call the round function at some point,
    # and we can't do that if it's overwritten somewhere!

    def __init__(self, rnd, step, input, decrypt=False, key=None):

        # check types of mandatory variables
        if type(rnd) != int:
            raise TypeError("Invalid type for 'round' parameter! "
                + "Expected int but got {}!".format(type(rnd)))

        if type(step) != str:
            raise TypeError("Invalid type for 'step' parameter! "
                + "Expected str but got {}".format(type(step)))

        if type(input) != np.ndarray:
            raise TypeError("Invalid type for 'input' parameter! "
                + "Expected numpy.ndarray but got {}!".format(type(input)))

        assert input.dtype.type is np.uint8, "Invalid input type! {}".format(input.dtype)
        if key is not None:
            assert key.dtype.type is np.uint8, "Invalid key type!"

        # make sure we get a key for the AddRoundKey step!
        if key is None and step == 'AddRoundKey':
            raise ValueError("You must pass a key for the AddRoundKey step!")

        # make sure we get a key for the FinalAddRoundKey step!
        if key is None and step == 'FinalAddRoundKey':
            raise ValueError("You must pass a key for the FinalAddRoundKey step!")

        # make sure we get a key for the InvAddRoundKey step!
        if key is None and step == 'InvAddRoundKey':
            raise ValueError("You must pass a key for the InvAddRoundKey step!")

        # the round of AES this datum belongs to
        # e.g. 1-14 for 256-bit AES
        self._round = rnd

        # the step that produced this datum
        # e.g. AddRoundKey, SubBytes, etc.
        self._step = step

        # remember wether we're an enryption- or a decryption-intermediate
        self._decrypt = decrypt

        # compute the datum
        if not self._decrypt:
            if step == 'ShiftRows':
                self._datum = AES_helper.cShiftRows(input)
            elif step == 'SubBytes':
                self._datum = AES_helper.cSubBytes(input)
            elif step == 'MixColumns':
                self._datum = AES_helper.cMixColumns(input)
            # AddRoundKey and FinalAddRoundKey are the same operation;
            # we give them different names to be unambiguous;
            # FinalAddRoundKey is another AddRoundKey step after
            # all rounds have been completed and returns the full
            # AES encrypted ciphertext.
            elif step == 'AddRoundKey' or step == 'FinalAddRoundKey':
                if len(key.shape) == 2:
                    self._datum = AES_helper.cAddRoundKey_multikey(input, key)
                else:
                    self._datum = AES_helper.cAddRoundKey(input, key)
            # nothing valid was passed, so raise an error:
            else:
                raise ValueError("Invalid step: was {} but expected one of:".format(step)
                    + " 'AddRoundKey', 'ShiftRows', 'SubBytes', 'MixColumns'!")

        else:
            # decryption functions:
            if step == 'ShiftRows':
                self._datum = AES_helper.cInvShiftRows(input)
            elif step == 'SubBytes':
                self._datum = AES_helper.cInvSubBytes(input)
            elif step == 'MixColumns':
                self._datum = AES_helper.cInvMixColumns(input)
            elif step == 'AddRoundKey' or step == 'FinalAddRoundKey':
                if len(key.shape) == 2:
                    self._datum = AES_helper.cInvAddRoundKey_multikey(input, key)
                else:
                    self._datum = AES_helper.cInvAddRoundKey(input, key)

            # nothing valid was passed, so raise an error:
            else:
                raise ValueError("Invalid step: was {} but expected one of:".format(step)
                    + " 'AddRoundKey', 'ShiftRows', 'SubBytes', 'MixColumns'!")

        assert self._datum.dtype.type is np.uint8, "Invalid type!"

        logger.debug("Successfully instantiated AESCryptoDatum"
            + " for round {} and step {}.".format(self._round, self._step))

    def __str__(self):
        """ Return a string of the format "<round>:<step>", as is used
            throughout this module to identify cryptographic data.
            E.g.: if self._round is 5 and self._step is "MixColumns",
            we return "5:MixColumns".
        """

        return str(self._round) + ':' + self._step

    @staticmethod
    def calc_from_value(step, input, key=None):
        """ Static method to simply calculate the output
            of the specified step, given the input and the optional key.
            Even though the 'FinalAddRoundKey' step exists, this function
            does not support it, as it is simply an AddRoundKey operation!
        """

        if step == 'SubBytes':
            return AES_helper.cSubBytes(input)
        elif step == 'ShiftRows':
            return AES_helper.cShiftRows(input)
        elif step == 'MixColumns':
            return AES_helper.cMixColumns(input)
        elif step == 'AddRoundKey' and key is not None:
            return AES_helper.cAddRoundKey(input, key)
        elif step == 'AddRoundKey' and key is None:
            raise ValueError("AddRoundKey step requires a key! key was: None")
        else:
            raise ValueError("Invalid step!")

class AESCryptoDataProcessor(CryptoDataProcessor):
    """ The AES-specific Crypto Data Processor.
        Works for all versions of AES, i.e. 128-, 192-
        and 256-bit.
    """

    # Please note:
    # AES consists of several rounds; variables used in this class
    # that represent a round are usually named 'rnd', because
    # 'round' is a built-in function of Python, and we want to prevent
    # errors related to this!
    # We may also want/need to call the round function at some point,
    # and we can't do that if it's overwritten somewhere!

    algo_name = 'AES'

    # global dictionary to specify valid number of rounds depending on keysize
    valid_rounds = {
        128: 10,
        192: 12,
        256: 14
    }

    def __init__(self, trace_file, **kwargs):

        # if the key in our trace file is a local field, it means
        # that each trace has its own unique key, so we set multikey to True.
        if trace_file.has_local_field('key'):
            self._multikey = True
        # if it's not a local field (but a global field), we just have
        # the one key in the trace file, so we set multikey to False
        else:
            self._multikey = False

        # TODO: fetch/compute keysize from key or pass to __init__
        # for now, only keysize 128 is supported.
        self._keysize = 128

        # check for validity of parameters
        if self._keysize not in [128, 192, 256]:
            raise ValueError("Invalid keysize! Got {}".format(self._keysize)
                             + " but expected one of 128, 192, 256!")

        # initialize the data we're saving as an empty dictionary
        self._data = {}

        # initialize an empty dictionary to store any guessed data in,
        # so we don't have to calculate guessed data multiple times
        self.guessed_data = {}

        # initialize an empty (None) guess key to keep track of
        # wether a new guess can make use of cached data or not
        self.guess_key = None

        # remember wether this processor was created for encryption
        # or decryption
        self._decrypt = kwargs.pop('decrypt', False)

        # 'decrypt'-parameter needs to be a boolean, of course
        if type(self._decrypt) != bool:
            raise TypeError("Invalid decrypt parameter! Expected bool but"
                            " got {}!".format(type(decrypt)))

        # set the number of rounds according to the keysize
        ## this is important, because there is a difference between
        ## 128-, 192- and 256-bit AES
        self.rounds = self.valid_rounds[self._keysize]

        # save the tracefile
        self.trace_file = trace_file

        # the steps AES consists of; constant for every version of AES
        self.steps = ['AddRoundKey',
                      'SubBytes',
                      'ShiftRows',
                      'MixColumns',
                      'FinalAddRoundKey',
                      # this isn't saved as a CryptoDatum, but can be
                      # fetched with __getitem__ just the same
                      'RoundKey']

        # initialize trace range to use;
        # we assume that the processor is called upon with the same
        # trace range several times; thus we can compute intermediates
        # faster by omitting data not within the trace range.
        # Intermediates will have to be re-computed if it changes, though.
        self._trange = None

        # once we're done with everything else, check wether kwargs
        # still contains unused parameters, and show a warning if it does
        if len(kwargs) != 0:
            logger.warning("Received unrecognized parameters, ignoring...")

    def __check_target(self, target):
        """ Check wether a given AESCryptoTarget object is valid for this
            processor.
        """

        rnd = target.rnd
        step = target.step

        if step not in self.steps:
            return False

        if rnd <= 0 or rnd > self.rounds:
            return False

        if target.decrypt != self._decrypt:
            return False

        return True


    def __normalize_datum(self, datum):
        """ Normalize the given datum into a tuple.
            Check wether the datum is a valid input in the first place.
        """

        # see if we received a AESCryptoTarget:
        if isinstance(datum, AESCryptoTarget):
            return datum.rnd, datum.step

        # split along the colon, as the valid format is
        # '<round>:<step>', e.g. '8:SubBytes'
        try:
            args = datum.split(':')
        # catch an AttributeError, which is thrown when we call split() on a
        # non-string type, and raise a TypeError instead. We do this because the
        # underlying problem isn't that type(datum) doesn't have an attribute
        # 'split()', but that datum has the wrong type to begin with!
        except AttributeError:
            raise TypeError("Invalid type! Expected str but got {} instead!"
                            .format(type(datum)))

        # if we end up with any number of string other than two
        # after the split, the datum was definitely not formatted
        # correctly!
        if len(args) != 2:
            raise ValueError("Invalid format of datum; expected '<round>:<step>',"
                             " got '{}'' instead!".format(datum))

        try:
            rnd = int(args[0])
            step = args[1]
        # except the ValueError so we can throw a custom ValueError with
        # a more precise error message!
        except ValueError:
            raise ValueError("Invalid format of datum; expected '<round>:<step>',"
                             " got '{}' instead!".format(datum))
        
        # round must be somewhere between 1 and the maximum number of rounds
        # (inclusive), for example for 256-bit AES: 1 through 14, _including_ 14!
        # but: if round is negative, we can let it 'overflow',
        # e.g. round -1 would be the last round, -2 second-to-last, and so on!
        if rnd not in range(1, self.rounds + 1):
            if rnd < 0:
                # the (rnd + 1) looks weird, but consider this:
                # rnd = -1
                # self.rounds = 10
                # self.rounds +(-1 + 1) = self.rounds
                # thus, -1 corresponds to the last round.
                rnd = self.rounds + (rnd + 1)
                # if rnd remains less than zero,
                # the input was too low
                if rnd <= 0:
                    raise IndexError("Invalid round! Expected value within "
                                     "1 and {}".format(self.rounds)
                                    +"! Got '{}'' instead!".format(rnd))
            elif rnd == self.rounds + 1 and step == 'RoundKey':
                return (rnd, step)
            else:
                raise IndexError("Invalid round! Expected value within "
                                 "1 and {}".format(self.rounds)
                                +"! Got '{}'' instead!".format(rnd))

        # if the step isn't equal to one of the steps we specified,
        # it's not a valid request.
        if step not in self.steps:
            raise IndexError("Invalid step! Expected one of {}".format(self.steps)
                + ", got '{}' instead!".format(step))

        # check more specific properties of the datum

        ## 'FinalAddRoundKey' step can only occur on last round
        ## (as it wouldn't be a full en-/decryption otherwise...)
        if rnd != self.rounds:
            if step == 'FinalAddRoundKey':
                raise IndexError("Invalid step or round! FinalAddRoundKey "
                                 "may only occur in final round!")

        ## last round of AES never contains the MixColumns step!
        if step == 'MixColumns':
            if self._decrypt and rnd == 1:
                raise IndexError("Invalid step or round! First round "
                                 "of decryption must not contain "
                                 "'MixColumns' step!")
            if not self._decrypt and rnd == self.rounds:
                raise IndexError("Invalid step or round! Last round "
                                 "of encryption must not contain "
                                 "'MixColumns' step!")

        # if we made it this far, we can be sure the request was valid!
        return (rnd, step)


    def provides(self, datum):
        """ Check wether this CPD can provide the given cryptographic
            datum.
            datum must be a string of the following format:
            '<round>:<step>', e.g. '9:AddRoundKey',
            or an instance of AESCryptoTarget.
        """

        # if we get a CryptoTarget, simply check if it's valid
        if isinstance(datum, AESCryptoTarget):
            return self.__check_target(datum)

        # else, try to normalize the datum (which should be a string)
        try:
            _, _ = self.__normalize_datum(datum)
        # if one of these errors is thrown, we know the datum is invalid,
        # and thus this processor can not provide it. So we return False.
        # It might be tempting to just do
        # except: return False
        # This can be dangerous, though, because it also catches
        # unexpected exceptions, like perhaps a MemoryError or something
        # else related to causes we might not be able to influence or predict!
        except (TypeError, ValueError, IndexError):
            return False

        # if nothing went wrong, return True
        return True

    def _fetch_input(self, target):
        """ Fetch the required input for the specified datum.
            This is the output of the previous datum,
            or the plaintext/ciphertext if it's the first round.
        """

        # this huge if-clause checks the validity of the request and
        # raises an appropriate exception or returns the appropriate
        # datum, depending on a number of factors.

        rnd, step = target.rnd, target.step

        ## the last round of AES _must not_ contain the 'MixColumns' step!
        if rnd == self.rounds and step == 'MixColumns' and not self._decrypt:
            raise ValueError("Bad request: 'MixColumns' does not occur in"
                             " last round of AES!")
        ## when decrypting, first round must not containt 'MixColumns' step!
        if step == 'MixColumns' and rnd == 1 and self._decrypt:
            raise ValueError("Bad request: 'MixColumns' does not occur"
                             " in first round of decryption!")
        ## only the last round of AES is allowed to contain
        # 'FinalAddRoundKey' step
        if step == 'FinalAddRoundKey' and rnd != self.rounds:
            raise ValueError("Bad request: 'FinalAddRoundKey' can only occur in"
                             " last round of AES!")

        # handle special cases where plaintext or ciphertext is required
        if not self._decrypt:
            # encryption steps...
            if step == 'AddRoundKey' and rnd == 1:
                ret = self.trace_file['plaintext', self._trange]
                assert ret.dtype.type is np.uint8
                return ret
        else: # self._decrypt == True
            # decryption steps...
            if step == 'AddRoundKey' and rnd == 1:
                ret = self.trace_file['ciphertext', self._trange]
                assert ret.dtype.type is np.uint8
                return ret

        # if we got this far, we can just do this:
        return self[self.get_previous_target(target), None, None]

    def __get_decryption_key_idx(self, rnd):
        """ Compute which key from the key schedule is needed for the
            specified round.
        """

        return self.rounds - (rnd - 1)

    def __get_decryption_key_rnd(self, rnd):
        """ Compute which round-key is needed for the corresponding
            decryption-round.
        """

        return self.__get_decryption_key_idx(rnd) + 1

    def _compute(self, target, key=None):
        """ Compute the requested datum.
            We assume that _compute is only called
            if the datum hasn't already been computed.
        """
        
        # fetch round and step
        rnd, step = target.rnd, target.step

        # fetch input for computation of our datum
        # this will automatically calculate all required data up to this datum,
        # but not more
        inpt = self._fetch_input(target)#.astype(np.uint8)
        assert inpt.dtype.type is np.uint8,\
            "Invalid inpt type! {}".format(inpt.dtype)

        # only set the key to something else if it's required,
        # i.e. if the step is 'AddRoundKey' (or 'FinalAddRoundKey')!
        if key is None and step == 'AddRoundKey':
            if not self._decrypt:
                if rnd == 1:
                    # fetch the key we need
                    key = self.trace_file['key', self._trange]\
                          if self._multikey\
                          else self.trace_file['key'].reshape(1,16)
                    # compute the next key we're going to need
                    self._curr_key = AES_helper.csubkey(key, 2, 128)
                else:
                    # fetch the 'current' key...
                    key = self._curr_key
                    # ... and update it for the next round
                    self._curr_key = AES_helper.csubkey(self._curr_key,
                                                              rnd + 1,
                                                              128)
            else: # self._decrypt == True
                if rnd == 1:
                    # fetch the key from the trace file
                    key = self.trace_file['key', self._trange]\
                          if self._multikey\
                          else self.trace_file['key'].reshape(1,16)
                    # transform it until we have the key we need
                    # (the last key in the schedule in this case)
                    for i in range(2, self.__get_decryption_key_rnd(2) + 2):
                        key = AES_helper.csubkey(key, i, 128)
                        # save the second-to-last key as the one we'll use next
                        if i == self.__get_decryption_key_rnd(2):
                            self._curr_key = key
                else:
                    # fetch the 'current' key...
                    key = self._curr_key
                    # ... and compute the next 'current' key from the
                    # key in the trace file...
                    self._curr_key = self.trace_file['key', self._trange]\
                                     if self._multikey\
                                     else self.trace_file['key'].reshape(1,16)
                    for i in range\
                        (2, self.__get_decryption_key_rnd(rnd + 1) + 1):
                        self._curr_key = AES_helper.csubkey(
                                                        self._curr_key,
                                                        i,
                                                        128)

        elif key is None and step == 'FinalAddRoundKey':
            if not self._decrypt:
                # fetch the current key; do not re-compute it, as it's
                # the last one. If we need an earlier datum, it will be
                # reset to the key in the trace file anyway.
                key = self._curr_key
            else: # self._decrypt == True
                # for decryption, the key is just the one saved in the file
                key = self.trace_file['key', self._trange]\
                      if self._multikey\
                      else self.trace_file['key']

        # set the datum in our data dictionary by creating a new,
        # appropriate CryptoDatum object.
        # NOTE: because the target implements a hash function,
        # it can safely be used as a key of a dictionary.
        if not self._multikey and key is not None:
            key = key.reshape(16)
        self._data[target] = AESCryptoDatum(rnd, step, inpt,
                                           decrypt=self._decrypt,
                                           key=key)

    def __getitem__(self, key):
        """ Return the requested datum;
            If needed, compute it first.
            Raise an exception if the requested datum is invalid.
            key must be a tuple with exactly three elements:
            the first must be a AESCryptoTarget,
            the second and third are the trace- and data-range respectively.
        """

        if not isinstance(key, tuple):
            raise TypeError("Bad input: "
                            "'key'-parameter must be a tuple of length 3!")

        if len(key) != 3:
            raise ValueError("Bad input: "
                            "'key'-parameter must be a tuple of length 3!")

        if not isinstance(key[0], AESCryptoTarget):
            raise ValueError("Bad input: "
                            "first item of 'key' must be "
                            "an instance of AESCryptoTarget!")

        target = key[0]
        trace_range = key[1]
        data_range = key[2]

        if target.keysize != self._keysize or target.decrypt != self._decrypt:
            raise ValueError("Target not compatible with this processor! "
                             "Keysize or decrypt mismatch!")

        # if a different trace range from the one we saved is given,
        # clear our data!
        if trace_range is not None and self._trange != trace_range:
            self._data = {}

        # process different possibilities for trace_range
        if trace_range is slice(None):
            # if the range is None, assume we want everything
            self._trange = slice(0, len(self.trace_file))
        elif type(trace_range) is int:
            # if it's an integer, make a slice that represents only
            # this one integer.
            self._trange = slice(trace_range, trace_range + 1)
        elif type(trace_range) is slice:
            # if it's a slice, we're good to go
            self._trange = trace_range
        elif trace_range is None:
            # trace_range will only be None if called from our own _compute;
            # the TraceFileHandler will only ever use slice(None) if the slice
            # is empty. If we get a None for trace_range, we don't need to worry,
            # as the trace_range will already have been applied somewhere
            # (again, since only this processor will call with None)
            pass
        else:
            raise TypeError("Invalid trace range!")

        # try to find the requested datum in our dictionary of data
        try:
            if data_range is None:
                return self._data[target].datum
            else:
                return self._data[target].datum[:,data_range]
        # catch the KeyError that occurs if it's not found...
        except KeyError:
            # ...and try to compute it...
            try:
                self._compute(target)
            # ...and try to return the round key if we catch a ValueError;
            # this is because compute will call CryptoDatum, which will
            # raise a ValueError, as long as __normalize_datum worked
            # on the datum; this specific behaviour should only occur
            # if the step is 'RoundKey'
            except ValueError:
                rnd, step = target.rnd, target.step
                if step == 'RoundKey':
                    return self.get_subkey(rnd)
                else:
                    raise KeyError("Item not provided by this processor!")
            # ...and raise a KeyError if that goes wrong, too!
            # note: we only except these three types of errors,
            # because we want to allow unexpected errors to bubble up!
            # if we did 'except: ...', even unexpected exceptions like, say,
            # MemoryError would be caught, and we don't want that to happen!
            except (TypeError, IndexError):
                raise KeyError("Item not provided by this processor!")

        # if the computation finished without throwing exceptions,
        # we can safely assume the datum is now saved in our _data dictionary,
        # and return it, while taking the data_range into account!
        ret = self._data[target]

        # in the interest of being able to process (very) large files,
        # we clear our data and only save the one datum we just computed;
        # this is usually fine since the data will be requested in order,
        # and thus computation won't be slowed down.
        # If an earlier datum is requested, we lose some speed, but usually
        # the trace range will also be changed for requests to earlier data,
        # meaning we would have to re-compute anyway.
        # Once a scaffold-global caching module exists, we might use that.
        self._data = {}

        # re-save the current datum; looks this way, but whatever...
        self._data[target] = ret

        # apply data_range (or not)
        if data_range is not None:
            ret = ret.datum[:,data_range]
        else:
            ret = ret.datum

        # done!
        return ret

    def get_crypto_target(self, datum):
        """ Returns a AESCryptoTarget object holding information about
            the datum, which does not include the computed datum,
            but some of its properties.
        """

        # if the datum is actually a CryptoTarget, we check its validity
        # and return it again.
        if isinstance(datum, AESCryptoTarget):
            if self.__check_target(datum):
                return datum
            else:
                raise ValueError("Invalid target for this processor!")

        # otherwise, assume we got a string, and compute a CryptoTarget
        # from that (right after normalizing it, of course).
        rnd, step = self.__normalize_datum(datum)
        ret = AESCryptoTarget(
                    rnd,
                    step,
                    self._keysize,
                    # self.get_previous_datum(datum),
                    decrypt=self._decrypt
              )

        if self.__check_target(ret):
            return ret
        else:
            raise ValueError("Invalid target for this processor!")

    def get_previous_target(self, target):
        """ Return the target that occurs before the one specified.
        """

        if target.decrypt != self._decrypt:
            raise ValueError("Incompatible target! Decrypt mismatch!")
        if target.keysize != self._keysize:
            raise ValueError("Incompatible target! Keysize mismatch!")

        rnd, step = None, None

        if self._decrypt:
            if target.step == 'AddRoundKey':
                if target.rnd == 1:
                    raise ValueError("No previous target")
                else:
                    rnd = target.rnd - 1
                    step = 'SubBytes'
            elif target.step == 'ShiftRows':
                rnd = target.rnd
                step = 'AddRoundKey' if target.rnd == 1 else 'MixColumns'
            elif target.step == 'SubBytes':
                rnd = target.rnd
                step = 'ShiftRows'
            elif target.step == 'MixColumns':
                rnd = target.rnd
                step = 'AddRoundKey'
            elif target.step == 'FinalAddRoundKey':
                rnd = target.rnd
                step = 'SubBytes'
        else:
            if target.step == 'AddRoundKey':
                if target.rnd == 1:
                    raise ValueError("No previous target")
                else:
                    rnd = target.rnd - 1
                    step = 'MixColumns'
            elif target.step == 'SubBytes':
                rnd = target.rnd
                step = 'AddRoundKey'
            elif target.step == 'ShiftRows':
                rnd = target.rnd
                step = 'SubBytes'
            elif target.step == 'MixColumns':
                rnd = target.rnd
                step = 'ShiftRows'
            elif target.step == 'FinalAddRoundKey':
                rnd = target.rnd
                step = 'ShiftRows'

        if rnd is None or step is None:
            raise ValueError("Invalid target!")

        return AESCryptoTarget(rnd, step, self._keysize, self._decrypt)

    def get_previous_datum(self, datum):
        """ Return the datum that occurs before the one specified.
        """

        rnd, step = self.__normalize_datum(datum)
        target = self.get_previous_target(
                            AESCryptoTarget(
                                rnd,
                                step,
                                self._keysize,
                                self._decrypt
                                )
                            )
        # since CryptoTargets provide a __str__() method, we can just
        # convert the target object to a string.
        return str(target)

    def get_subkey(self, rnd):
        """ Return the sub key from the key schedule for the specified round.
            rnd may be between 1 and self.rounds + 1 because
            there are self.rounds + 1 different subkeys; one for each round
            plus one for the final AddRoundKey step! rnd may also be negative,
            just like in Python's list indexing.
        """

        # enable wrap-around of index
        if rnd < 0:
            rnd = self.rounds + (rnd + 2)

        assert rnd > 0 and rnd <= self.rounds + 1, "Invalid round!"

        # start with the key in our trace file
        curr_key = self.trace_file['key', self._trange]\
                   if self._multikey\
                   else self.trace_file['key'].reshape(1,16)

        # if we're decrypting, we need to fix the round we consider!
        if self._decrypt:
            rnd = self.__get_decryption_key_rnd(rnd)

        # if the round is one, that's already the right subkey!
        if rnd == 1:
            return curr_key

        # otherwise, keep computing the next subkey until we have
        # the one we need!
        for i in range(2, rnd + 1):
            curr_key = AES_helper.csubkey(curr_key, i, 128)

        # shape the key back from (1,16) to (16) if we're not multikey
        if not self._multikey:
            curr_key = curr_key.reshape(16)

        return curr_key

    def __str__(self):
        """ Return a string containing some useful information about
            an instance of this class.
            This function is used by print() to determine what to print.
        """

        ret = "AESCryptoDataProcessor; "
        ret += "keysize: " + str(self._keysize) + "; "
        if len(self._data) > 0:
            ret += str(len(self._data)) + " already computed data point(s)"
        else:
            ret += "no data computed yet!"

        return ret

    @property
    def crypto_targets(self):
    # def list_fields(self):
        """ Return a list of all valid fields this processor provides. """

        ret = []
        if self._decrypt:
            for step in ['AddRoundKey', 'ShiftRows', 'SubBytes']:
                ret.append('1:' + step)
            for i in range(2, self.rounds + 1):
                for step in ['AddRoundKey', 'MixColumns',
                             'ShiftRows', 'SubBytes']:
                    ret.append(str(i) + ':' + step)
        else:
            for i in range(self.rounds):
                for step in ['AddRoundKey', 'SubBytes',
                             'ShiftRows', 'MixColumns']:
                    ret.append(str(i) + ':' + step)
            for step in ['AddRoundKey', 'SubBytes', 'ShiftRows']:
                ret.append('10:' + step)
        ret.append('10:FinalAddRoundKey')

        return ret

    @staticmethod
    def get_key_schedule(key):
        """ Return the key schedule for a specified key.
        """

        return AES_helper.ckey_schedule(key)

    @staticmethod
    def get_datum_string(rnd, step):
        """ Simple function to build a valid datum-identifying string
            from an integer specifying the round and a string specifying
            the step.
        """

        assert type(rnd) is int, "Invalid round type!"
        assert type(step) is str, "Invalid step type!"

        return str(rnd) + ':' + step

    def guess_data(self, target, guess, trace_range, data_range=None):
        """ Calculate intermediate(s) based on a guessed key instead
            of the actual key stored in the trace file.
        """

        # discard any previously computed data
        # this is inefficient, but with large sets of data
        # we probably can't afford to keep all of it, and with
        # small sets of data it won't take long to recompute.
        self._data = {}

        # if target is a string, create a CryptoTarget from it
        # if it's already a target, validate it.
        target = self.get_crypto_target(target)

        # set trace range
        self._trange = trace_range

        # create a target to compute the very first intermediate
        # we do this separately so that _compute doesn't attempt
        # to fetch the key from the trace file
        # every subsequent key will depend on the _curr_key,
        # which we set manually further below
        ark1_target = AESCryptoTarget(1, 'AddRoundKey',
                                      self._keysize, self._decrypt)

        temp_guess = guess
        if target.decrypt:
            for i in range(2, self.rounds + 2):
                temp_guess = AES_helper.csubkey(temp_guess.reshape(1,16), i, 128)

        # compute the first intermediate
        self._compute(ark1_target, key=temp_guess)

        # set the curr_key variable to the next key;
        # it was ignored in the _compute call just now because we passed
        # a key explicitly. Now, though, we must set it to something sensible
        self._curr_key = AES_helper.csubkey(guess.reshape(1,16), 2, 128)
        if target.decrypt:
            for i in range(3, self.rounds + 1):
                self._curr_key = AES_helper.csubkey(self._curr_key, i, 128)

        # compute the remaining intermediates if necessary,
        # and fetch our final result
        ret = self[target, self._trange, data_range]

        # clear data in case we get a regular, non-guess request for
        # the same datum with the same trace range later.
        self._data = {}

        # ... and return
        return ret

    @staticmethod
    def full_encryption(tracefile, trace_range, keysize, tracefilehandler=None):
        """ perform a full encryption, which is to compute all intermediates,
            and then perform one last AddRoundKey on the final intermediate,
            thus creating a fully AES-encrypted result.
        """

        from scaffold.core import TraceFileHandler

        if keysize not in [128,192,256]:
            raise ValueError("Invalid keysize!")

        if tracefilehandler is None:
            tfh = TraceFileHandler(tracefile, algo='AES')

        num_rnds = {128:10, 192:12, 256:14}[keysize]

        return tfh[str(num_rnds) + ':FinalAddRoundKey', trace_range]
