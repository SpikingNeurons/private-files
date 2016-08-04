# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
import cProfile
# Code to generate html file and compile pyd file
import subprocess
import pyximport
subprocess.call(['cython', '-a', 'SEED_helper.pyx'])
pyximport.install(setup_args={'include_dirs': [np.get_include(), '.']},
                  inplace=True)

_SCAFFOLD = False
_PRINT = False

from collections import namedtuple

# from Cython.Includes.libcpp.stack import stack

if _SCAFFOLD:
    from scaffold.crypto.SEED import SEEDCryptoTarget
    import scaffold.crypto.SEED_helper as SEED_helper
    from scaffold.core import TraceFile, TraceFileHandler, config
    import scaffold
else:
    import SEED_helper

import time, os, sys
import unittest

# some global params
if _SCAFFOLD:
    cfg = config.Config(scaffold.config_file())
    SAMPLES_PATH = cfg.get('file_settings', 'samples_dir')
    TRACE_FILE_SINGLE_KEY = 'aes8bit_fixed_key_10k.bin'
    TRACE_FILE_MULTI_KEY = 'aes8bit_fixed_key_10k.bin'
    _G_file_name_single_key = os.path.join(SAMPLES_PATH, TRACE_FILE_SINGLE_KEY)
    _G_file_name_multi_key = os.path.join(SAMPLES_PATH, TRACE_FILE_MULTI_KEY)
_G_custom_stream = sys.stdout

# Test vectors to check SEED encryption and decryption
test_vector = namedtuple('test_vector', 'key, ptx, ctx')

test_vectors_sksp = [
    test_vector(
        key=np.asarray([[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                        ], np.uint8),
        ptx=np.asarray([[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
                        ], np.uint8),
        ctx=np.asarray([[0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68,
                         0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
                        ], np.uint8),
        ptx=np.asarray([[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                        ], np.uint8),
        ctx=np.asarray([[0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50,
                         0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8,
                         0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85]
                        ], np.uint8),
        ptx=np.asarray([[0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9,
                         0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D]
                        ], np.uint8),
        ctx=np.asarray([[0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D,
                         0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
                         0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]
                        ], np.uint8),
        ptx=np.asarray([[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                         0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]
                        ], np.uint8),
        ctx=np.asarray([[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                         0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]
                        ], np.uint8)
    )
]

test_vector_mkmp = test_vector(
    key=np.asarray([
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        [0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8,
         0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85],
        [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
         0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]
    ], np.uint8),
    ptx=np.asarray([
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        [0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9,
         0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D],
        [0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
         0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]
    ], np.uint8),
    ctx=np.asarray([
        [0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68,
         0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB],
        [0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50,
         0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43],
        [0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D,
         0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A],
        [0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
         0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]
    ], np.uint8)
)


def get_test_vector_big_skmp(test_big_size):
    return test_vector(
        key=np.asarray(
            [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
             0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
            np.uint8),
        ptx=np.repeat(
            np.asarray(
                [[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                  0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ctx=np.repeat(
            np.asarray(
                [[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                  0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]],
                np.uint8
            ),
            test_big_size, axis=0)
    )


def get_test_vector_big_mkmp(test_big_size):
    return test_vector(
        key=np.repeat(
            np.asarray(
                [[0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
                  0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ptx=np.repeat(
            np.asarray(
                [[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                  0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ctx=np.repeat(
            np.asarray(
                [[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                  0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]],
                np.uint8
            ),
            test_big_size, axis=0)
    )


def print_utility(time_taken, data_size, method_name):
    if not _PRINT:
        return
    _G_custom_stream.write('\n..... ' + str(time_taken) + '\t sec taken by ' +
                           method_name + ' for data size of ' +
                           str(data_size))
    _G_custom_stream.write('\n..... ')
    pass


def write_file(data, filename):
    with open(filename, "w") as f:
        f.write(data)


def _test_c_types():
    aa = np.arange(500).reshape(100, 5)
    print(aa[:, 0].flags['C_CONTIGUOUS'])
    print(aa[0, :].flags['C_CONTIGUOUS'])

    tt = aa[:, 0]
    _print_pointer(tt)
    tt = aa[:, 1]
    _print_pointer(tt)
    tt = aa[0, :]
    _print_pointer(tt)
    tt = aa[1, :]
    _print_pointer(tt)

    print(aa.ctypes.shape)
    print(aa.ctypes.strides)
    print(aa.ctypes.strides)


def _print_hex(arr):
    fill = 0
    if arr.dtype == np.uint8:
        fill = 2
    elif arr.dtype == np.uint16:
        fill = 4
    elif arr.dtype == np.uint32:
        fill = 8
    elif arr.dtype == np.uint64:
        fill = 16
    for elem in arr:
        ss = hex(elem)[2:].zfill(fill)
        print(ss, end=' ')
    print('\n')


def _print_pointer(data):
    import ctypes
    import sys
    print('.........................................')
    print(data.ctypes.data_as(ctypes.c_void_p))
    print('Size: ' + str(sys.getsizeof(data)))
    print('dtype: ' + str(data.dtype))
    print('shape: ' + str(data.shape))
    print('strides: ' + str(data.strides))
    print('')
    print('')
    print('')


def _print_extra_info(time_taken):
    """
    generates string to be printed
    """
    return '\n\t>>>> 1111 : ' + str(1111) + \
           '\n....' + 'time taken: [[' + str(time_taken) + ']]' + \
           '\n'


def _third_party_encrypt_seed(plain_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.SEED(key),
        modes.ECB(),
        backend
    )
    encryptor = cipher.encryptor()
    ct = encryptor.update(plain_text)
    ct += encryptor.finalize()
    return ct


def _third_party_decrypt_seed(cipher_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.SEED(key),
        modes.ECB(),
        backend
    )
    decryptor = cipher.decryptor()
    ct = decryptor.update(cipher_text)
    ct += decryptor.finalize()
    return ct


def _third_party_encrypt_aes(plain_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend
    )
    encryptor = cipher.encryptor()
    ct = encryptor.update(plain_text)
    ct += encryptor.finalize()
    return ct


def _third_party_decrypt_aes(cipher_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend
    )
    decryptor = cipher.decryptor()
    ct = decryptor.update(cipher_text)
    ct += decryptor.finalize()
    return ct


def third_party_encrypt(plain_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_encrypt_seed(
            plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_encrypt_aes(
            plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


def third_party_decrypt(cipher_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_decrypt_seed(
            cipher_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_decrypt_aes(
            cipher_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


class TimingTest(unittest.TestCase):
    def setUp(self):
        self.big_size = 10000
        self.threads_to_use = 1
        pass

    def tearDown(self):
        pass

    def test_SEED_helper_encrypt_multiple_key(self):
        """
        Check for cython implementation encryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        vec = get_test_vector_big_mkmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=vec.ptx,
            keys=vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_multiple_key failed')

    def test_SEED_helper_decrypt_multiple_key(self):
        """
        Check for cython implementation decryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        vec = get_test_vector_big_mkmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=vec.ctx,
            keys=vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_multiple_key failed')

    def test_SEED_helper_encrypt_single_key(self):
        """
        Check for cython implementation encryption of SEED (with single keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        test_vec = get_test_vector_big_skmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=test_vec.ptx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, test_vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_single_key failed')

    def test_SEED_helper_decrypt_single_key(self):
        """
        Check for cython implementation decryption of SEED (with single keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        test_vec = get_test_vector_big_skmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=test_vec.ctx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, test_vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_single_key failed')


class StandAloneTest(unittest.TestCase):

    def test_third_party_SEED_encrypt(self):
        """
        Check for third party encryption of SEED
        """
        try:
            equality_check = True
            for ptx, ctx, key in zip(
                    test_vector_mkmp.ptx,
                    test_vector_mkmp.ctx,
                    test_vector_mkmp.key):
                ctx_ret = third_party_encrypt(ptx, key, 'SEED')
                equality_check = \
                    equality_check and np.array_equal(ctx, ctx_ret)
            self.assertTrue(
                equality_check, 'Third party SEED encryption failed')
        except ImportError as e:
            self.skipTest('Third party SEED library missing: ' + str(e))

    def test_third_party_SEED_decrypt(self):
        """
        Check for third party decryption of SEED
        """
        try:
            equality_check = True
            for ptx, ctx, key in zip(
                    test_vector_mkmp.ptx,
                    test_vector_mkmp.ctx,
                    test_vector_mkmp.key):
                ptx_ret = third_party_decrypt(ctx, key, 'SEED')
                equality_check = \
                    equality_check and np.array_equal(ptx, ptx_ret)
            self.assertTrue(
                equality_check, 'Third party SEED decryption failed')
        except ImportError as e:
            self.skipTest('Third party SEED library missing: ' + str(e))

    def test_SEED_helper_encrypt_multiple_key(self):
        """
        Check for cython implementation encryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        res = cy_solver.execute(
            val_text=test_vector_mkmp.ptx,
            keys=test_vector_mkmp.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vector_mkmp.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_multiple_key failed')

    def test_SEED_helper_decrypt_multiple_key(self):
        """
        Check for cython implementation decryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        res = cy_solver.execute(
            val_text=test_vector_mkmp.ctx,
            keys=test_vector_mkmp.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vector_mkmp.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_multiple_key failed')

    def test_SEED_helper_encrypt_single_key(self):
        """
        Check for cython implementation encryption of SEED (with single keys)
        """
        test_vec = get_test_vector_big_skmp(10)
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        res = cy_solver.execute(
            val_text=test_vec.ptx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_single_key failed')

    def test_SEED_helper_decrypt_single_key(self):
        """
        Check for cython implementation decryption of SEED (with single keys)
        """
        test_vec = get_test_vector_big_skmp(10)
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        res = cy_solver.execute(
            val_text=test_vec.ctx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_single_key failed')

    def test_key_schedule_while_encryption(self):
        """
        Test of key schedule generation logic while encryption
        """
        pass

    def test_key_schedule_while_decryption(self):
        """
        Test of key schedule generation logic while decryption
        """
        pass

    def test_intermediate_F_while_encryption(self):
        """
        We only test output of function F which is intermediate value.
        We check for all rounds.
        """
        pass

    def test_intermediate_F_while_decryption(self):
        """
        We only test output of function F which is intermediate value.
        We check for all rounds.
        """
        pass


class TraceFileHandlerTest(unittest.TestCase):

    def setUp(self):
        if _SCAFFOLD:
            self.skipTest('No scaffold project found')

    def test_encrypt_with_CryptoDataTarget(self):
        """
        Check encryption with TraceFileHandler (with single key) and
        CryptoDataTarget
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': False})
        key = tf['key'].astype(np.uint8)
        ptx = tfh['plaintext'].astype(np.uint8)

        # we are using fake trace file so generate cipher text
        # from third party code
        ctx = []
        for p in ptx:
            c = third_party_encrypt(p, key, 'SEED')
            ctx.append(c)
        ctx = np.asarray(ctx)

        # use TraceFileHandler to get the output of final round
        target = SEEDCryptoTarget(
            trace_file=tf, rnd=16, step='Output', keysize=128, decrypt=False)
        res = tfh[target].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ctx, res)
        self.assertTrue(
            equality_check, 'test_encrypt_single_key_with_tfh failed')

    def test_decrypt_with_CryptoDataTarget(self):
        """
        Check decryption with TraceFileHandler (with single key) and
        CryptoDataTarget
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and cipher text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': True})
        key = tf['key'].astype(np.uint8)
        ctx = tfh['ciphertext'].astype(np.uint8)

        # we are using fake trace file so generate
        # plain text from third party code
        ptx = []
        for c in ctx:
            p = third_party_decrypt(c, key, 'SEED')
            ptx.append(p)
        ptx = np.asarray(ptx)

        # use TraceFileHandler to get the output of final round
        target = SEEDCryptoTarget(
            trace_file=tf, rnd=16, step='Output', keysize=128, decrypt=True)
        res = tfh[target].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ptx, res)
        self.assertTrue(
            equality_check, 'test_decrypt_single_key_with_tfh failed')

    def test_encrypt_single_key_with_tfh(self):
        """
        Check encryption with TraceFileHandler (with single key)
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': False})
        key = tf['key'].astype(np.uint8)
        ptx = tfh['plaintext'].astype(np.uint8)

        # we are using fake trace file so
        # generate cipher text from third party code
        ctx = []
        for p in ptx:
            c = third_party_encrypt(p, key, 'SEED')
            ctx.append(c)
        ctx = np.asarray(ctx)

        # use TraceFileHandler to get the output of final round
        res = tfh['16:Output'].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ctx, res)
        self.assertTrue(
            equality_check, 'test_encrypt_single_key_with_tfh failed')

    def test_decrypt_single_key_with_tfh(self):
        """
        Check decryption with TraceFileHandler (with single key)
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and cipher text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': True})
        key = tf['key'].astype(np.uint8)
        ctx = tfh['ciphertext'].astype(np.uint8)

        # we are using fake trace file so
        # generate plain text from third party code
        ptx = []
        for c in ctx:
            p = third_party_decrypt(c, key, 'SEED')
            ptx.append(p)
        ptx = np.asarray(ptx)

        # use TraceFileHandler to get the output of final round
        res = tfh['16:Output'].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ptx, res)
        self.assertTrue(
            equality_check, 'test_decrypt_single_key_with_tfh failed')

    def test_encrypt_multi_key_with_tfh(self):
        """
        Check encryption with TraceFileHandler (with multi key)
        """
        # get the trace file
        tf = TraceFile(_G_file_name_multi_key)

        # check trace file
        if not tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with multi key')
            return

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': False})
        key = tf['key'].astype(np.uint8)
        ptx = tfh['plaintext'].astype(np.uint8)

        # we are using fake trace file so
        # generate cipher text from third party code
        ctx = []
        for p, k in zip(ptx, key):
            c = third_party_encrypt(p, k, 'SEED')
            ctx.append(c)
        ctx = np.asarray(ctx)

        # use TraceFileHandler to get the output of final round
        res = tfh['16:Output'].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ctx, res)
        self.assertTrue(
            equality_check, 'test_encrypt_multi_key_with_tfh failed')

    def test_decrypt_multi_key_with_tfh(self):
        """
        Check decryption with TraceFileHandler (with multi key)
        """
        # get the trace file
        tf = TraceFile(_G_file_name_multi_key)

        # check trace file
        if not tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with multi key')
            return

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': True})
        key = tf['key'].astype(np.uint8)
        ctx = tfh['ciphertext'].astype(np.uint8)

        # we are using fake trace file so generate
        # plain text from third party code
        ptx = []
        for c, k in zip(ctx, key):
            p = third_party_decrypt(c, k, 'SEED')
            ptx.append(p)
        ptx = np.asarray(ptx)

        # use TraceFileHandler to get the output of final round
        res = tfh['16:Output'].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ptx, res)
        self.assertTrue(
            equality_check, 'test_decrypt_multi_key_with_tfh failed')

    def test_decrypt_steps_provided(self):
        """
        Test to check the steps provided by SEED algorithm in decrypt mode.
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': True})

        # get crypto data processor
        ret = tfh._data_processor.crypto_targets

        # unittest check
        for r in ret:
            rnd = int(r.split(':')[0])
            step = r.split(':')[1]
            if not 0 < rnd <= 16:
                self.assertTrue(False, 'Round not in range.')
            if step not in [x[0] for x in
                            SEED_helper.STEPS_PROVIDED_AND_MAX_RANGE]:
                self.assertTrue(False, 'Step: ' + step + ' not provided')
            if step == [x[0] for x in
                            SEED_helper.STEPS_PROVIDED_AND_MAX_RANGE][-1]:
                if rnd is not 16:
                    self.assertTrue(
                        False,
                        'Step: ' + step + ' is only available for round 16'
                    )

        pass

    def test_encrypt_steps_provided(self):
        """
        Test to check the steps provided by SEED algorithm in encrypt mode.
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': False})

        # get crypto data processor
        ret = tfh._data_processor.crypto_targets

        # unittest check
        for r in ret:
            rnd = int(r.split(':')[0])
            step = r.split(':')[1]
            if not 0 < rnd <= 16:
                self.assertTrue(False, 'Round not in range.')
            if step not in [x[0] for x in
                            SEED_helper.STEPS_PROVIDED_AND_MAX_RANGE]:
                self.assertTrue(False, 'Step: ' + step + ' not provided')
            if step == [x[0] for x in
                            SEED_helper.STEPS_PROVIDED_AND_MAX_RANGE][-1]:
                if rnd is not 16:
                    self.assertTrue(
                        False,
                        'Step: ' + step + ' is only available for round 16'
                    )

        pass


class CryptoDataTargetTest(unittest.TestCase):

    def setUp(self):
        if _SCAFFOLD:
            self.skipTest('No scaffold project found')

    def test_encrypt_with_CryptoDataTarget(self):
        """
        Check encryption with TraceFileHandler (with single key) and
        CryptoDataTarget
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and plain text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': False})
        key = tf['key'].astype(np.uint8)
        ptx = tfh['plaintext'].astype(np.uint8)

        # we are using fake trace file so generate cipher text
        # from third party code
        ctx = []
        for p in ptx:
            c = third_party_encrypt(p, key, 'SEED')
            ctx.append(c)
        ctx = np.asarray(ctx)

        # use TraceFileHandler to get the output of final round
        target = SEEDCryptoTarget(
            trace_file=tf, rnd=16, step='Output', keysize=128, decrypt=False)
        res = tfh[target].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ctx, res)
        self.assertTrue(
            equality_check, 'test_encrypt_single_key_with_tfh failed')

    def test_decrypt_with_CryptoDataTarget(self):
        """
        Check decryption with TraceFileHandler (with single key) and
        CryptoDataTarget
        """
        # get the trace file
        tf = TraceFile(_G_file_name_single_key)

        # check trace file
        if tf.has_local_field('key'):
            self.skipTest('You did not provide trace file with single key')
            return

        # get key and cipher text
        tfh = TraceFileHandler(tf, algo='SEED', algo_param={'decrypt': True})
        key = tf['key'].astype(np.uint8)
        ctx = tfh['ciphertext'].astype(np.uint8)

        # we are using fake trace file so generate
        # plain text from third party code
        ptx = []
        for c in ctx:
            p = third_party_decrypt(c, key, 'SEED')
            ptx.append(p)
        ptx = np.asarray(ptx)

        # use TraceFileHandler to get the output of final round
        target = SEEDCryptoTarget(
            trace_file=tf, rnd=16, step='Output', keysize=128, decrypt=True)
        res = tfh[target].astype(np.uint8)

        # equality_check
        equality_check = np.array_equal(ptx, res)
        self.assertTrue(
            equality_check, 'test_decrypt_single_key_with_tfh failed')


def all_tests():
    test_loader = unittest.TestLoader()
    suite1 = test_loader.loadTestsFromTestCase(TimingTest)
    suite2 = test_loader.loadTestsFromTestCase(StandAloneTest)
    # suite = unittest.TestSuite()
    # suite.addTest(suite1)
    # suite.addTest(suite2)
    _G_custom_stream.write('\n\n\n|||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||||||\n\n')
    s = unittest.TextTestRunner(stream=_G_custom_stream, descriptions=True,
                                verbosity=3).run(suite1)
    _G_custom_stream.write('\n|||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||\n')
    _G_custom_stream.write('|||||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||\n\n')
    s = unittest.TextTestRunner(stream=_G_custom_stream, descriptions=True,
                                verbosity=3).run(suite2)
    _G_custom_stream.write('\n|||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||\n\n')
    pass


def one_test():
    pass


if __name__ == '__main__':
    #unittest.main()
    all_tests()
