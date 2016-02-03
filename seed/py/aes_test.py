#!python
# -*- coding: utf-8 -*-
"""
A simple module with several test functions for the new AES implementation,
designed to work well with the new AESCryptoDataProcessor.
Simply do:
$ python test.py [arguments]
from a command line or
$ run test [arguments]
from an ipython session to run a test.
Valid arguments:
    full
    round
    rounds
    schedule
    large
    ARK
    SR
    SB
    MC
Specifying no argument will run a full test, same as 'full'.
The 'rounds' argument expects a second argument: an integer that specifies
the number of rounds.
"""

import time
from scaffold.core import TraceFile, TraceFileHandler, config
from scaffold.crypto import AES, AES_helper, AESCryptoTarget
import numpy as np
import scaffold, os, sys

cfg = config.Config(scaffold.config_file())
SAMPLES_PATH = cfg.get('file_settings', 'samples_dir')

T_FILE = 'aes8bit_fixed_key_10k.bin'
# T_FILE = 'rkey/job0000.bin'

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

def test_ARK():
    """ compute only the AddRoundKey step of AES and compare it to an
        already working implementation.
    """

    print("AddRoundKey")
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    ark_res = tfh['1:AddRoundKey'].astype(np.uint8)
    # same thing
    # ark_res = AES.AddRoundKey(tf['plaintext'], tf['key'])

    print("my implementation:")
    print_hex(ark_res[0])
    print("old implementation:")
    other_res = AES.AES_round(tf['plaintext'][0], tf['key'], False)
    print_hex(other_res[0])

def test_invARK():
    """ compare AddRoundKey to it's inverse """

    key = np.array([i for i in range(0x0f, -0x01, -1)], dtype=np.uint8)
    inpt = np.array([[i for i in range(0x10)]], dtype=np.uint8)
    print("input:")
    print_hex(inpt[0])
    print("key:")
    print_hex(key)
    print("AddRoundKey transformation:")
    out = AES.AddRoundKey(inpt, key)
    print_hex(out[0])
    print("Inverse AddRoundKey transformation (should equal input):")
    out = AES.InvAddRoundKey(out, key)
    print_hex(out[0])

def test_SR():
    """ compute only the ShiftRows step of AES and compare it to an
        already working implementation.
    """

    print("ShiftRows")
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    sr_res = tfh['1:ShiftRows'].astype(np.uint8)

    print("my implementation:")
    print_hex(sr_res[0])
    print("old implementation:")
    other_res = AES.AES_round(tf['plaintext'][0], tf['key'], False)
    print_hex(other_res[1])

def test_invSR():
    """ compare shift rows to inverse shift rows """

    inpt = np.array([[i for i in range(0x10)]], dtype=np.uint8)
    print("input:")
    print_hex(inpt[0])
    print("ShiftRows transformation:")
    out = AES.ShiftRows(inpt)
    print_hex(out[0])
    print("Inverse SubBytes transformation (should equal input):")
    out = AES.InvShiftRows(out)
    print_hex(out[0])

def test_SB():
    """ compute only the SubBytes step of AES and compare it to an
        already working implementation.
    """

    print("SubBytes")
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    sb_res = tfh['1:SubBytes'].astype(np.uint8)

    print("my implementation:")
    print_hex(sb_res[0])
    print("old implementation:")
    other_res = AES.AES_round(tf['plaintext'][0], tf['key'], False)
    print_hex(other_res[2])

def test_invSB():
    """ compare subbytes with inverse subbytes"""

    inpt = np.array([[i for i in range(0x10)]], dtype=np.uint8)
    print("input:")
    print_hex(inpt[0])
    print("SubBytes transformation:")
    out = AES.SubBytes(inpt)
    print_hex(out[0])
    print("Inverse SubBytes transformation (should equal input):")
    out = AES.InvSubBytes(out)
    print_hex(out[0])

def test_MC():
    """ compute only the MixColumns step of AES and compare it to an
        already working implementation.
    """

    print("MixColumns")
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    mc_res = tfh['1:MixColumns'].astype(np.uint8)

    print_hex(mc_res[0])

def test_invMC():
    """ compare mix columns to inverse mix columns
    """

    # inpt = np.array([[0x01 for i in range(16)]], dtype=np.uint8)
    inpt = np.array([[0xdb, 0x13, 0x53, 0x45,
                      0xf2, 0x0a, 0x22, 0x5c,
                      0x01, 0x01, 0x01, 0x01,
                      0xd4, 0xd4, 0xd4, 0xd5]], dtype=np.uint8)
    print("input:")
    print_hex(inpt[0])
    out = AES.MixColumns(inpt)
    print("MixColumns output:")
    print_hex(out[0])
    print("Inverse MixColumns of that (should equal input):")
    out = AES.InvMixColumns(out)
    print_hex(out[0])

def test_round():
    """ compute one full round of AES and compare it to an
        already working implementation.
    """
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    print("Comparing implementations for first round...\n")

    my_res = tfh['1:MixColumns'].astype(np.uint8)

    print("ciphertext encrypted by me:")
    print_hex(my_res[0])

    other_res = AES.AES_round(tf['plaintext'][0], tf['key'], False)

    print("ciphertext encrypted by old implementation:")
    print_hex(other_res[4])

def test_rounds(n):
    """ compute the nth round of AES and compare it to an already
        working implementation.
    """

    assert n <= 10, "number of rounds must be less than 10!"

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    _key_schedule = AES.key_schedule(tf['key'])
    inpt = tf['plaintext'][0]

    for i in range(1, n + 1):
        if i < 10:
            new_res = tfh[str(i) + ':MixColumns'].astype(np.uint8)[0]
        else:
            new_res = tfh[str(i) + ':ShiftRows'].astype(np.uint8)[0]
        old_res = AES.AES_round(inpt, _key_schedule[i - 1], i == 10)
        inpt = old_res[4]

        for a,b in zip(new_res, old_res[4]):
            print(a == b, end=' ')
        print('\n')
        if i == 10:
            print_hex(new_res)
            print_hex(old_res[4])

def test_full(verbose=True):
    """ do a full run of AES and compare it to a working implementation.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    if verbose:
        print("Complete AES128 result:\n")
    res = tfh["10:FinalAddRoundKey"].astype(np.uint8)

    if verbose:
        print("ciphertext encrypted by me (sample):")
        print_hex(res[0])
        print_hex(res[1])
        print_hex(res[2])
        print("ciphertext saved in tracefile (sample):")
        print_hex(tf['ciphertext'][0])
        print_hex(tf['ciphertext'][1])
        print_hex(tf['ciphertext'][2])

    if np.all(np.equal(res, tf['ciphertext'])):
        return True
    else:
        return False

def test_time():
    """ do a full run of AES and time it, without wasting time
        by computing a reference result.
    """

    print("doing full AES run on tracefile...")
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128')

    tic = time.clock()
    res = tfh["10:FinalAddRoundKey"]

    print("done!")
    print("time taken:", time.clock() - tic, "seconds")

    print()
    print("Computed results (sample):")
    print_hex(res[0].astype(np.uint8))
    print_hex(res[1].astype(np.uint8))
    print_hex(res[2].astype(np.uint8))

def test_large_file(verbose=True):
    """ test the AESCryptoDataProcessor on a very large trace file.
    """

    path = "rkey/job0"

    if verbose:
        print("Generating large TraceFile object...")
        sys.stdout.flush()
    # generate a large trace file object
    filenames = []
    for i in range(316):
        if i < 10:
            filenames.append(os.path.join(SAMPLES_PATH, path + '00' + str(i) + ".bin"))
        elif i < 100:
            filenames.append(os.path.join(SAMPLES_PATH, path + '0' + str(i) + ".bin"))
        else:
            filenames.append(os.path.join(SAMPLES_PATH, path + str(i) + ".bin"))

    tf = TraceFile(*filenames)
    # tf = TraceFile(os.path.join(SAMPLES_PATH, 'rkey/job0000.bin'))

    if verbose:
        print("done generating!")
        print("creating TraceFileHandler object...")
        sys.stdout.flush()
    tfh = TraceFileHandler(tf, algo='AES128')
    if verbose:
        print("created TraceFileHandler!")
        print("Running full AES encryption...")
        sys.stdout.flush()
    # do some random other requests to see if data is cleared properly!
    res = tfh['8:MixColumns', 5:70]
    res = tfh['1:ShiftRows', 5:70]

    # do the genuine request and time it
    tic = time.clock()
    res = tfh['10:FinalAddRoundKey', 1:18]

    if verbose:
        print("done!")
        print("time taken:", time.clock() - tic, "seconds")
        print("\n")
        print("result shape:")
        print(res.shape)
        print("Computed results (sample):")
        print_hex(res[0].astype(np.uint8))
        print_hex(res[1].astype(np.uint8))
        print("...")
        print_hex(res[-1].astype(np.uint8))
        print()
        print("Saved results (sample):")
        print_hex(tf['ciphertext', 1:18][0])
        print_hex(tf['ciphertext', 1:18][1])
        print("...")
        print_hex(tf['ciphertext', 1:18][-1])
        print("\nKey schedule (sample):")
        for i in range(11):
            print_hex(tfh._data_processor._key_schedule[0][i])

        print("Shape of key-schedule:")
        print(tfh._data_processor._key_schedule.shape, "\n")

        print("Key in trace file (sample):")
        print_hex(tf['key'][0])

    if np.all(np.equal(res, tf['ciphertext', 1:18])):
        return True
    else:
        return False

def test_lol():
    path = "rkey/job0"

    print("Generating large TraceFile object...")
    sys.stdout.flush()
    # generate a large trace file object
    filenames = []
    for i in range(316):
        if i < 10:
            filenames.append(os.path.join(SAMPLES_PATH, path + '00' + str(i) + ".bin"))
        elif i < 100:
            filenames.append(os.path.join(SAMPLES_PATH, path + '0' + str(i) + ".bin"))
        else:
            filenames.append(os.path.join(SAMPLES_PATH, path + str(i) + ".bin"))

    tf = TraceFile(*filenames)
    # tf = TraceFile(os.path.join(SAMPLES_PATH, 'rkey/job0000.bin'))

    print("done generating!")
    print("creating TraceFileHandler object with single key...")
    sys.stdout.flush()
    tfh = TraceFileHandler(tf, algo='AES128', algo_param={'key':tf['key'][0]})
    print("created TraceFileHandler!")
    print("timing encryption...")
    tic = time.clock()
    res = tfh['10:FinalAddRoundKey']
    print("time taken:", time.clock() - tic, "seconds")

def test_key_schedule(verbose=True):
    """ compute the full key schedule of AES for the basic key that
        is simply 16 0-bytes. Outcome can be found here:
        http://www.samiam.org/key-schedule.html
    """

    ksize = 16
    if verbose:
        print("see http://www.samiam.org/key-schedule.html for test vectors!\n")
        print("keyschedule computed for keysize {}:".format(ksize))

    key = np.array([0x00 for i in range(ksize)], dtype=np.uint8)
    # key = np.array([0x2b, 0x7e, 0x15, 0x16,
    #                 0x28, 0xae, 0xd2, 0xa6,
    #                 0xab, 0xf7, 0x15, 0x88,
    #                 0x09, 0xcf, 0x4f, 0x3c],
    #                 dtype=np.uint8)
    # key = np.array([0x9a, 0x91, 0x30, 0x35, 0xf5, 0x7c, 0xb2, 0xea,
    #                 0xfa, 0x94, 0xc0, 0xf5, 0xb4, 0xac, 0x8a, 0x69],
    #                 dtype=np.uint8)
    res = AES_helper.ckey_schedule(key)

    if verbose:
        for row in res:
            print_hex(row)

        print()

    reference =\
        np.array([
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x62, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63],
            [0x9b, 0x98, 0x98, 0xc9, 0xf9, 0xfb, 0xfb, 0xaa, 0x9b, 0x98, 0x98, 0xc9, 0xf9, 0xfb, 0xfb, 0xaa],
            [0x90, 0x97, 0x34, 0x50, 0x69, 0x6c, 0xcf, 0xfa, 0xf2, 0xf4, 0x57, 0x33, 0x0b, 0x0f, 0xac, 0x99],
            [0xee, 0x06, 0xda, 0x7b, 0x87, 0x6a, 0x15, 0x81, 0x75, 0x9e, 0x42, 0xb2, 0x7e, 0x91, 0xee, 0x2b],
            [0x7f, 0x2e, 0x2b, 0x88, 0xf8, 0x44, 0x3e, 0x09, 0x8d, 0xda, 0x7c, 0xbb, 0xf3, 0x4b, 0x92, 0x90],
            [0xec, 0x61, 0x4b, 0x85, 0x14, 0x25, 0x75, 0x8c, 0x99, 0xff, 0x09, 0x37, 0x6a, 0xb4, 0x9b, 0xa7],
            [0x21, 0x75, 0x17, 0x87, 0x35, 0x50, 0x62, 0x0b, 0xac, 0xaf, 0x6b, 0x3c, 0xc6, 0x1b, 0xf0, 0x9b],
            [0x0e, 0xf9, 0x03, 0x33, 0x3b, 0xa9, 0x61, 0x38, 0x97, 0x06, 0x0a, 0x04, 0x51, 0x1d, 0xfa, 0x9f],
            [0xb1, 0xd4, 0xd8, 0xe2, 0x8a, 0x7d, 0xb9, 0xda, 0x1d, 0x7b, 0xb3, 0xde, 0x4c, 0x66, 0x49, 0x41],
            [0xb4, 0xef, 0x5b, 0xcb, 0x3e, 0x92, 0xe2, 0x11, 0x23, 0xe9, 0x51, 0xcf, 0x6f, 0x8f, 0x18, 0x8e]
        ], dtype=np.uint8)

    if np.all(np.equal(res, reference)):
        return True
    else:
        return False

def test_old_key_schedule():
    """ compute the key schedule from the old implementation;
        should return same results as test_key_schedule().
    """

    key = np.array([0x00 for i in range(16)])

    res = []
    for i in range(1, 11):
        key = AES.Key_schedule(key, i)
        res.append(key)

    for row in res:
        for col in row:
            s = hex(col)[2:]
            if len(s) == 1:
                s = '0' + s
            print(s, end=' ')
        print()

def test_decrypt(verbose=True):
    """ compute decryption intermediates and compare the computed
        plaintext to the one saved in the file.
    """

    if verbose:
        print("Testing decryption functionality...")
        print("Loading trace file...")
        sys.stdout.flush()
    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES128', algo_param={'decrypt':True})

    # print("key schedule shape:")
    # print(tfh._data_processor._key_schedule.shape)

    if verbose:
        print("Computing decryption intermediates...")
        sys.stdout.flush()

    tic = time.clock()

    res = tfh["10:FinalAddRoundKey"]

    tic = time.clock() - tic

    if verbose:
        print("Done!")
        print("Time taken:", tic, "seconds!")
        print()

        print("result shape:", res.shape)
        print("Computed plaintext (sample):")
        print_hex(res[0].astype(np.uint8))
        print_hex(res[1].astype(np.uint8))
        print_hex(res[2].astype(np.uint8))
        print("...")
        print_hex(res[-1].astype(np.uint8))

        print()
        print("plaintext shape:", tf['plaintext'].shape)
        print("Saved plaintext (sample):")
        print_hex(tf['plaintext'][0].astype(np.uint8))
        print_hex(tf['plaintext'][1].astype(np.uint8))
        print_hex(tf['plaintext'][2].astype(np.uint8))
        print("...")
        print_hex(tf['plaintext'][-1].astype(np.uint8))

    if np.all(np.equal(res, tf['plaintext'])):
        if verbose:
            print("PASS")
        return True
    else:
        if verbose:
            print("FAIL")
        return False
    # print("keys")
    # print(tfh._data_processor._key_schedule.shape)
    # print_hex(tfh._data_processor._key[0])
    # print("final key used:")
    # print_hex(tfh._data_processor._key_schedule[-1])
    # print("key schedule:")
    # for row in tfh._data_processor._key_schedule[0]:
    #     print_hex(row)

def test_large_decrypt(verbose=True):
    """ Decrypt a large file.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, 'rkey/job0000.bin'))
    tfh = TraceFileHandler(tf, algo='AES', algo_param={'decrypt':True})

    tic = time.clock()

    # (use a random slice just to see if trace range works)
    res = tfh["10:FinalAddRoundKey", 2:21]

    tic = time.clock() - tic

    if verbose:
        print("Done!")
        print("Time taken:", tic, "seconds!")
        print()

        print("result shape:", res.shape)
        print("Computed plaintext (sample):")
        print_hex(res[0].astype(np.uint8))
        print_hex(res[1].astype(np.uint8))
        print_hex(res[2].astype(np.uint8))
        print("...")
        print_hex(res[20].astype(np.uint8))

        print()
        print("plaintext shape:", tf['plaintext'].shape)
        print("Saved plaintext (sample):")
        print_hex(tf['plaintext'][0].astype(np.uint8))
        print_hex(tf['plaintext'][1].astype(np.uint8))
        print_hex(tf['plaintext'][2].astype(np.uint8))
        print("...")
        print_hex(tf['plaintext'][20].astype(np.uint8))

    # (also apply that random slice to the plaintext to compare...)
    if np.all(np.equal(res, tf['plaintext'][2:21])):
        if verbose:
            print("PASS")
        return True
    else:
        if verbose:
            print("FAIL")
        return False

def test_compare_decrypt():
    """ Compare encryption- and decryption-intermediates.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh_enc = TraceFileHandler(tf, algo='AES')
    tfh_dec = TraceFileHandler(tf, algo='AES', algo_param={'decrypt':True})

    print("1st ARK enc / last ARK dec")
    print_hex(tfh_enc['1:AddRoundKey'][0].astype(np.uint8))
    print_hex(tfh_dec['10:FinalAddRoundKey'][0].astype(np.uint8))

def test_subkey_generation():
    """ Test new subkey function in AES_helper.
    """

    test_key_schedule()
    curr_key = np.array([0x2b, 0x7e, 0x15, 0x16,
                         0x28, 0xae, 0xd2, 0xa6,
                         0xab, 0xf7, 0x15, 0x88,
                         0x09, 0xcf, 0x4f, 0x3c],
                        dtype=np.uint8)
    print("subkeys")
    print_hex(curr_key)
    for i in range(2,12):
        curr_key = AES_helper.csubkey(curr_key, i, 128)
        print_hex(curr_key)

def test_target_getting(verbose=True):
    """ Test new behaviour of accepting CryptoTargets in addition to
        strings as __getitem__'s key.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES')

    target = AESCryptoTarget(10, 'FinalAddRoundKey', 128, False)
    res = tfh[target]

    if np.all(np.equal(res, tf['ciphertext'])):
        return True
    else:
        return False

def test_guess(verbose=True):
    """ Test the guess_data functionality.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, start=0, stop=10,
                           algo='AES', algo_param={'decrypt':False})

    # guess = np.array([0x00 for i in range(16)], dtype=np.uint8)
    # we set the guess-key as the actual key so we can easily compare results
    guess = {'key': tf['key']}

    # compute some random intermediates
    # res = tfh['8:MixColumns']
    # res = tfh['4:ShiftRows']
    # res = tfh['2:SubBytes']

    # guess some random intermediates
    # res = tfh.guess_data('5:SubBytes', guess)
    # res = tfh.guess_data('3:MixColumns', guess)

    # compute a guess which we can easily check for validity...
    # res = tfh.guess_data('1:AddRoundKey', guess)
    res = tfh.guess_data('10:FinalAddRoundKey', guess)

    # ...and check its validity
    # if np.all(np.equal(np.bitwise_xor(guess, tf['plaintext'][0]), res[0])):
    if verbose:
        print_hex(res[0])
        print_hex(tfh['10:FinalAddRoundKey', 0:10][0].astype(np.uint8))
    if np.all(np.equal(res, tfh['10:FinalAddRoundKey', 0:10])):
        return True
    else:
        return False

def test_guess_decrypt(verbose=True):
    """ Test the guess_data functionality while decrypting.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, start=0, stop=10,
                           algo='AES', algo_param={'decrypt':True})

    guess = {'key': tf['key']}

    res = tfh.guess_data('10:FinalAddRoundKey', guess)
    if verbose:
        print_hex(res[0])
        print_hex(tfh['10:FinalAddRoundKey', 0:10][0].astype(np.uint8))

    if np.all(np.equal(res, tfh['10:FinalAddRoundKey', 0:10])):
        return True
    else:
        return False

def test_subkey(verbose=True):
    """ Test getting of subkeys.
    """

    tf = TraceFile(os.path.join(SAMPLES_PATH, T_FILE))
    tfh = TraceFileHandler(tf, algo='AES')

    rk_target = AESCryptoTarget(1, 'RoundKey', 128, False)

    res = tfh[rk_target]

    if np.all(np.equal(res, tf['key'])):
        return True
    else:
        return False

def test_all():
    """ Run all important tests and show results in a concise manner.
    """

    total = 0

    if test_full(verbose=False):
        print("Full run: PASS")
        total += 1
    else:
        print("Full run: FAIL")
    sys.stdout.flush()

    if test_decrypt(verbose=False):
        print("Decryption run: PASS")
        total += 1
    else:
        print("Decryption run: FAIL")
    sys.stdout.flush()

    if test_large_file(verbose=False):
        print("Multi-key run: PASS")
        total += 1
    else:
        print("Multi-key run: FAIL")
    sys.stdout.flush()

    if test_large_decrypt(verbose=False):
        print("Multi-key decrypt run: PASS")
        total += 1
    else:
        print("Multi-key decrypt run: FAIL")
    sys.stdout.flush()

    if test_target_getting(verbose=False):
        print("Using target-object: PASS")
        total += 1
    else:
        print("Using target-object: FAIL")
    sys.stdout.flush()

    if test_guess(verbose=False):
        print("Guessing a key: PASS")
        total += 1
    else:
        print("Guessing a key: FAIL")
    sys.stdout.flush()

    if test_guess_decrypt(verbose=False):
        print("Guessing a key (decrypting): PASS")
        total += 1
    else:
        print("Guessing a key (decrypting): FAIL")
    sys.stdout.flush()

    if test_subkey(verbose=False):
        print("Fetching roundkey: PASS")
        total += 1
    else:
        print("Fetching roundkey: FAIL")

    print()
    print(str(total) + "/8 tests passed!")

if __name__ == '__main__':
    import sys
    import cProfile

    if len(sys.argv) == 1:
        test_all()
        sys.exit(0)

    # more than two arguments are always invalid
    if len(sys.argv) > 3:
        sys.exit(1)

    # parse the different possible arguments
    if sys.argv[1] == 'full':
        test_full()
        # cProfile.run("test_full()", sort='tottime')
    elif sys.argv[1] == 'round':
        test_round()
    elif sys.argv[1] == 'schedule':
        test_key_schedule()
    elif sys.argv[1] == 'ARK':
        test_ARK()
    elif sys.argv[1] == 'SR':
        test_SR()
    elif sys.argv[1] == 'SB':
        test_SB()
    elif sys.argv[1] == 'MC':
        test_MC()
    elif sys.argv[1] == 'time':
        test_time()
    elif sys.argv[1] == 'rounds':
        if sys.argv[2].isdigit():
            test_rounds(int(sys.argv[2]))
        else:
            sys.exit(1)
    elif sys.argv[1] == 'large':
        test_large_file()
        # cProfile.run("test_large_file()", sort='tottime')
    elif sys.argv[1] == 'lol':
        cProfile.run("test_lol()", sort='tottime')
    elif sys.argv[1] == 'decrypt':
        test_decrypt()
    elif sys.argv[1] == 'comp':
        test_compare_decrypt()
    elif sys.argv[1] == 'invmc':
        test_invMC()
    elif sys.argv[1] == 'invsb':
        test_invSB()
    elif sys.argv[1] == 'invsr':
        test_invSR()
    elif sys.argv[1] == 'invark':
        test_invARK()
    elif sys.argv[1] == 'subkey':
        test_subkey_generation()
    elif sys.argv[1] == 'largedec':
        test_large_decrypt()
    elif sys.argv[1] == 'guess':
        test_guess()
    elif sys.argv[1] == 'guess_dec':
        test_guess_decrypt()
    else:
        print("Invalid argument!")
        sys.exit(1)

    sys.exit(0)
